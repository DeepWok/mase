import sys
from numpy import outer
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from transformers import GPT2TokenizerFast
import transformers
import torch.nn.functional as F
import math
import numpy
sys.path.append('...')


def make_autoregressive_mask_for(x):
    length = x.size(1)
    ones = x.new_ones((length, length))
    mask = torch.triu(ones, diagonal=1) != 0.0
    return mask


def make_position_indices_for(x):
    length = x.size(1)
    batch_size = x.size(0)
    indices = torch.arange(length, device=x.device).repeat(batch_size, 1)
    return indices


def load_lookup_table(file, device):
    data = torch.from_numpy(numpy.genfromtxt(file, delimiter='\t')).float()
    levels = data.size(0)
    lower_bound = data[0,1].item()
    weight = data[:,1].unsqueeze(1).cuda(device)
    return weight, lower_bound, levels


def apply_lut_to_normalized(x, lut, bit_degredation=0):
    lut_weight, lut_lb, lut_levels = lut
    deg_factor = 2**bit_degredation
    x = x.mul(lut_levels - deg_factor).div(deg_factor).round().mul(deg_factor).to(dtype=torch.long)
    x = F.embedding(x, lut_weight).squeeze(-1)
    return x


class QuantizeValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quant_levels, min_val, max_val, quant_mode, lut_min=None):
        with torch.no_grad():
            diff = max_val - min_val
            x = x.clamp(min_val, max_val).add(-1.0 * min_val).div(diff + 1e-8).mul(quant_levels - 1)

            if quant_mode == 'det':
                x = x.round()
                x = x.div(quant_levels - 1).mul(diff).add(min_val)
            elif quant_mode == 'rand':
                x = x.add(torch.rand_like(x).add(-0.5)).round() # randn* 0.288 gives same std as 0-1 rand(), if want to use normal dist.
                x = x.div(quant_levels - 1).mul(diff).add(min_val)
            
            if lut_min is not None:
                pos_x = torch.relu(x)
                neg_x = x - pos_x
                lms = lut_min * max_val
                pos_x[pos_x < lms] = lms
                lms = lut_min * torch.abs(min_val)
                neg_x[neg_x > -lms] = -lms
                x = pos_x + neg_x

            return x

    @staticmethod
    def backward(ctx, grad_output):
        # STE
        return grad_output, None, None, None, None, None


class QuantizeStats(nn.Module):
    def __init__(self, percentile, use_clipping=True):
        super(QuantizeStats, self).__init__()
        self.register_buffer('running_min', torch.tensor(0.0))
        self.register_buffer('running_max', torch.tensor(0.0))
        self.max_calibration_steps = 1
        self.initial_calibration_steps = 0
        #self.register_buffer('calibration_done', torch.tensor(False))
        self.calibration_done = torch.tensor(False)
        self.activations = []
        self.percentile = percentile
        self.use_clipping = use_clipping

    def update(self, tensor):
        if self.use_clipping:
            if not self.calibration_done.item():
                self.initial_calibration_steps += 1
                finished = False

                if self.initial_calibration_steps >= self.max_calibration_steps:
                    finished = True
                    self.calibration_done = torch.tensor(True)

                with torch.no_grad():
                    self.activations.extend(tensor.detach().cpu().tolist())

                    if finished:
                        maximum = numpy.percentile(self.activations, self.percentile)
                        self.running_max = torch.tensor(maximum, device=tensor.device, dtype=tensor.dtype)
                        minimum = tensor.min()
                        minimum = minimum if minimum >= 0.0 else -maximum
                        self.running_min = torch.tensor(minimum, device=tensor.device, dtype=tensor.dtype)
                        self.activations.clear() # free the memory
                    else:
                        self.running_min = tensor.min()
                        self.running_max = tensor.max()
        
        else:
            alpha = 0.999
            with torch.no_grad():
                cur_min = tensor.min()
                cur_max = tensor.max()

                if self.initial_calibration_steps == 0:
                    self.initial_calibration_steps += 1
                    self.running_min = cur_min
                    self.running_max = cur_max
                else:
                    self.running_min = alpha * self.running_min + (1.0 - alpha) * cur_min
                    self.running_max = alpha * self.running_max + (1.0 - alpha) * cur_max



    def get(self):
        return self.running_min, self.running_max


def shot_noise_linear(w, x, n_photons_target, phone_lut=None, slm_lut=None, extract=False, extract_name=None):
    noise_level = 0.021

    if n_photons_target != 0:
        quantize = QuantizeValue.apply
        use_lut = (phone_lut is not None) and (slm_lut is not None)
        w_max = torch.max(w)
        w_norm = apply_lut_to_normalized(w / (1e-8 + w_max), slm_lut) if use_lut else w / (1e-8 + w_max)
        x_max = torch.max(x, dim=2, keepdim=True)[0]
        x_norm = apply_lut_to_normalized(x / (1e-8 + x_max), phone_lut, bit_degredation=0) if use_lut else x / (1e-8 + x_max)

        out_opt = F.linear(x_norm, w_norm, bias=None)
        photons_per_act = n_photons_target * x_norm.size(2) / (x_norm.sum(dim=2, keepdim=True) + 1e-8)
        fluence_Wx = out_opt * photons_per_act
        noise_Wx = torch.poisson(fluence_Wx)
        out = noise_Wx / photons_per_act

        random_noise = noise_level * out.mean()
        out = torch.normal(out, random_noise)

        out = x_max * out * w_max
    else:
        out = F.linear(x, w, bias=None)

    if extract and n_photons_target != 0:
        torch.save({'x': x_norm[1, :512, :].detach().clone(),
                    'w': w_norm[:512].detach().clone(),
                    'out': out_opt[1, :512, :512].detach().clone(),
                    'noise_level': noise_level},
                    #'noise_value': random_noise},
                    extract_name)

    return out


def shot_noise_bhmm(x, y, n_photons_target, phone_lut=None, slm_lut=None, extract=False, extract_name=None):
    # perform xy matrix-multiply like matrix vector, where matrix "slices" in y are like W and x is the vectors. Thus take max over whole matrices in y as we would for W
    noise_level = 0.0565

    if n_photons_target != 0:
        quantize = QuantizeValue.apply
        use_lut = (phone_lut is not None) and (slm_lut is not None)
        x_max = torch.max(x, dim=3, keepdim=True)[0]
        x_norm = apply_lut_to_normalized(x / (1e-8 + x_max), phone_lut, bit_degredation=0) if use_lut else x / (1e-8 + x_max)
        y_max = torch.amax(y, dim=(2, 3), keepdim=True)
        y_norm = apply_lut_to_normalized(y / (1e-8 + y_max), slm_lut) if use_lut else y / (1e-8 + y_max)

        out_opt = torch.matmul(x_norm, y_norm)
        photons_per_act = n_photons_target * x_norm.size(3) / (x_norm.sum(dim=3, keepdim=True) + 1e-8)
        fluence_mm = out_opt * photons_per_act
        noise_Wx = torch.poisson(fluence_mm)
        out = noise_Wx / photons_per_act

        random_noise = noise_level * out.mean()
        out = torch.normal(out, random_noise)

        out = x_max * out * y_max
    else:
        out = torch.matmul(x, y)

    if extract and n_photons_target != 0:
        torch.save({'x': x_norm[0, 0, :, :].detach().clone(),
                    'y': y_norm[0, 0, :, :].detach().clone(),
                    'out': out_opt[0, 0, :, :].detach().clone(),
                    'noise_level': noise_level},
                    #'noise_value': random_noise},
                    extract_name)

    return out


class QuantizedLinear(nn.Module):
    def __init__(self, in_feats, out_feats, use_noise=True):
        super(QuantizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(out_feats, in_feats))
        self.input_stats = QuantizeStats(99.99)
        self.output_stats = QuantizeStats(99.9999)
        nn.init.xavier_uniform_(self.weight)
        self.quantize = False
        self.photon_target = 0
        self.slm_lut = load_lookup_table('LUTs/SLM_AmpLUT.txt', device=torch.device('cuda:0'))
        self.phone_lut = load_lookup_table('LUTs/PhoneLUT.txt', device=torch.device('cuda:0'))
        self.use_lut = (self.slm_lut is not None) and (self.phone_lut is not None)
        if self.use_lut:
            _, self.slm_cutoff, _ = self.slm_lut
        else:
            self.slm_cutoff = None
        self.force_quantized_eval = False
        self.extract_simulated = False
        self.extract_name = ''
        self.noise = use_noise
        #print('L module using LUT: {}'.format(self.use_lut))

    def _weight_min(self):
        with torch.no_grad():
            return self.weight_min

    def _weight_max(self):
        with torch.no_grad():
            return self.weight_max

    def enable_quantization(self, clipping=True):
        with torch.no_grad():
            if clipping:
                weight_values = self.weight.detach().cpu().tolist()
                maximum = numpy.percentile(weight_values, 99).item()
                self.weight_max = torch.tensor(maximum, dtype=self.weight.dtype, device=self.weight.device)
                self.weight_min = torch.tensor(-maximum, dtype=self.weight.dtype, device=self.weight.device)
            else:
                self.weight_min = self.weight.min()
                self.weight_max = self.weight.max()
            self.quantize = True

    def set_photon_target(self, n_photons):
        self.photon_target = n_photons

    def forward(self, x):
        if self.quantize:
            quantize = QuantizeValue.apply
            if self.training or self.force_quantized_eval:
                # QAT for activations
                if self.training:
                    self.input_stats.update(x)
                input_min, input_max = self.input_stats.get()
                quantized_x = quantize(x, 256, input_min, input_max, 'det')
                quantized_weights = quantize(self.weight, 256, self._weight_min(), self._weight_max(), 'det', self.slm_cutoff) # 160
                out = F.linear(quantized_x, quantized_weights, bias=None)
                if self.training:
                    self.output_stats.update(out)
                output_min, output_max = self.output_stats.get()
                quantized_out = quantize(out, 256, output_min, output_max, 'rand')
                return quantized_out
            else:
                # shot noise simulation for linear layer, per-token
                input_min, input_max = self.input_stats.get()
                weight_min, weight_max = self._weight_min(), self._weight_max()

                if self.use_lut:
                    w = self.weight.clamp(weight_min, weight_max)
                    x = x.clamp(input_min, input_max)
                else:
                    quantize = QuantizeValue.apply
                    x = quantize(x, 256, input_min, input_max, 'det')
                    w = quantize(self.weight, 256, weight_min, weight_max, 'det', self.slm_cutoff)

                pos_x = F.relu(x)
                neg_x = torch.abs(x - pos_x)
                pos_w = F.relu(w)
                neg_w = torch.abs(w - pos_w)
                out = shot_noise_linear(pos_w, pos_x, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_0.pt') \
                    + shot_noise_linear(neg_w, neg_x, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_1.pt') \
                    - shot_noise_linear(pos_w, neg_x, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_2.pt') \
                    - shot_noise_linear(neg_w, pos_x, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_3.pt')

                output_min, output_max = self.output_stats.get()
                out = out.clamp(output_min, output_max)
                #out = quantize(out, 64, output_min, output_max, 'det')
                return out
        else:
            out = F.linear(x, self.weight, bias=None)
            return out


class QuantizedMatmul(nn.Module):
    def __init__(self):
        super(QuantizedMatmul, self).__init__()
        self.input1_stats = QuantizeStats(99.99)
        self.input2_stats = QuantizeStats(98)
        self.output_stats = QuantizeStats(99.9999)
        self.quantize = False
        self.photon_target = 0           
        self.slm_lut = load_lookup_table('LUTs/SLM_AmpLUT.txt', device=torch.device('cuda:0'))
        self.phone_lut = load_lookup_table('LUTs/PhoneLUT.txt', device=torch.device('cuda:0'))   
        self.use_lut = (self.slm_lut is not None) and (self.phone_lut is not None)
        if self.use_lut:
            _, self.slm_cutoff, _ = self.slm_lut
        else:
            self.slm_cutoff = None
        self.force_quantized_eval = False
        self.extract_simulated = False
        self.extract_name = ''
        #print('MM module using LUT: {}'.format(self.use_lut))

    def enable_quantization(self):
        self.quantize = True

    def set_photon_target(self, n_photons):
        self.photon_target = n_photons

    def forward(self, x, y):
        if self.quantize:
            quantize = QuantizeValue.apply
            if self.training or self.force_quantized_eval:
                # QAT for activations
                if self.training:
                    self.input1_stats.update(x)
                    self.input2_stats.update(y)
                x_min, x_max = self.input1_stats.get()
                y_min, y_max = self.input2_stats.get()
                xq = quantize(x, 256, x_min, x_max, 'det')
                yq = quantize(y, 256, y_min, y_max, 'det', self.slm_cutoff)
                out = torch.matmul(xq, yq)
                if self.training:
                    self.output_stats.update(out)
                out_min, out_max = self.output_stats.get()
                outq = quantize(out, 256, out_min, out_max, 'rand')
                return outq
            else:
                # Shot noise simulation for broadcasted matrix-matrix multiply
                x_min, x_max = self.input1_stats.get()
                y_min, y_max = self.input2_stats.get()

                if self.use_lut:
                    x = x.clamp(x_min, x_max)
                    y = y.clamp(y_min, y_max)
                else:
                    quantize = QuantizeValue.apply
                    x = quantize(x, 256, x_min, x_max, 'det')
                    y = quantize(y, 256, y_min, y_max, 'det', self.slm_cutoff)

                pos_x = F.relu(x)
                neg_x = torch.abs(x - pos_x)
                pos_y = F.relu(y)
                neg_y = torch.abs(y - pos_y)
                out = shot_noise_bhmm(pos_x, pos_y, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_0.pt') \
                    + shot_noise_bhmm(neg_x, neg_y, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_1.pt') \
                    - shot_noise_bhmm(pos_x, neg_y, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_2.pt') \
                    - shot_noise_bhmm(neg_x, pos_y, self.photon_target, self.phone_lut, self.slm_lut, self.extract_simulated, self.extract_name + '_3.pt')

                output_min, output_max = self.output_stats.get()
                out = out.clamp(output_min, output_max)
                #out = quantize(out, 64, output_min, output_max, 'det')
                return out
        else:
            out = torch.matmul(x, y)
            return out


class QuantizedMHA(nn.Module):
    def __init__(self, embed_dim, heads):
        super(QuantizedMHA, self).__init__()
        assert embed_dim % heads == 0
        self.n_heads = heads
        self.Wq = QuantizedLinear(embed_dim, embed_dim)
        self.Wk = QuantizedLinear(embed_dim, embed_dim)
        self.Wv = QuantizedLinear(embed_dim, embed_dim)
        self.qmm1 = QuantizedMatmul()
        self.dropout_wq = nn.Dropout(0.1)
        self.dropout_wk = nn.Dropout(0.1)
        self.dropout_wv = nn.Dropout(0.1)
        self.qmm2 = QuantizedMatmul()
        self.Wout = QuantizedLinear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask):
        b = x.size(0)
        n = x.size(1)
        h = self.n_heads
        d = x.size(2)

        def arrange_heads(acts):
            # incoming shape of b, n, d, want b, h, n, d/h
            return acts.view(b, n, h, -1).transpose(1, 2)

        q = arrange_heads(self.dropout_wq(self.Wq(x)))
        k = arrange_heads(self.dropout_wk(self.Wk(x)))
        v = arrange_heads(self.dropout_wv(self.Wv(x)))

        attn = self.qmm1(q, k.transpose(2, 3)) # yields b, h, n, n
        masked = attn.masked_fill(mask, float("-inf"))
        softmax_attn = self.dropout1(F.softmax(masked / math.sqrt(d // h), dim=3))
        out = self.qmm2(softmax_attn, v) # b, h, n, d/h

        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.dropout2(out)
        out = self.Wout(out)
        return out


class QuantizedFF(nn.Module):
    def __init__(self, embed_dim, expansion_dim):
        super(QuantizedFF, self).__init__()
        self.first_drop = nn.Dropout(0.1)
        self.layer1 = QuantizedLinear(embed_dim, expansion_dim, use_noise=True)
        self.act = nn.ReLU6(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.layer2 = QuantizedLinear(expansion_dim, embed_dim, use_noise=True)

    def forward(self, x):
        out = self.first_drop(x)
        out = self.layer1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out


class QuantizedDecoderLayer(nn.Module):
    def __init__(self, features, heads):
        super(QuantizedDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(features)
        self.attn = QuantizedMHA(features, heads)
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(features)
        self.ff = QuantizedFF(features, features * 4)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x, attn_mask):
        # no need for key mask for gpt; autoregressive masking already prevents 'real' tokens from attending to padding tokens to the right
        identity = x
        out = self.norm1(x)
        out = self.attn(out, attn_mask)
        out = self.drop1(out)
        out = out + identity
        identity = out
        out = self.norm2(out)
        out = self.ff(out)
        out = self.drop2(out)
        out = out + identity
        return out


class _QuantizedGPT(nn.Module):
    def __init__(self, features, heads, tokenizer, layers, max_length):
        super(_QuantizedGPT, self).__init__()
        vocab_size = len(tokenizer) + 8 - len(tokenizer) % 8 # pad vocab size to 8-multiple for tensor core acceleration
        assert vocab_size % 8 == 0
        self.pos_embedding = nn.Embedding(max_length, features)
        self.word_embedding = nn.Embedding(vocab_size, features, padding_idx = tokenizer.pad_token_id)
        self.embedding_dropout = nn.Dropout(0.1)
        self.decoders = nn.ModuleList([QuantizedDecoderLayer(features, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(features)
        self.output_head = nn.Linear(features, vocab_size)
        nn.init.normal_(self.word_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward_embedding(self, x):
        embedded = self.word_embedding(x)
        return embedded

    def forward_attn(self, x):
        mask = make_autoregressive_mask_for(x)
        pos = make_position_indices_for(x)
        pos_embed = self.embedding_dropout(self.pos_embedding(pos) + x)
        decoded = pos_embed
        for layer in self.decoders:
            decoded = layer(decoded, mask)
        
        out = self.norm(decoded)
        return out

    def forward(self, x):
        embedded = self.forward_embedding(x)
        decoded = self.forward_attn(embedded)
        out = self.output_head(decoded)
        return out


class QuantizedGPT(pl.LightningModule):
    def __init__(self, features, heads, layers=6, max_length=1024):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        self.transformer = _QuantizedGPT(features, heads, self.tokenizer, layers, max_length)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        self.lr = 0.0005
        self.photon_target = 0
        self.training_steps = 100000
        self.extracting = False
        self.use_adam = True

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, x):
        return self.transformer(x)

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        preds = self(xs)
        features = preds.size(2)
        preds = preds.view(-1, features)
        ys = ys.view(-1)
        loss = self.loss(preds, ys)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        preds = self(xs)
        features = preds.size(2)
        preds = preds.view(-1, features)
        ys = ys.view(-1)
        loss = self.loss(preds, ys)
        self.val_loss.update(loss)

    def validation_epoch_end(self, outputs):
        self.log('validation loss', self.val_loss)

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        preds = self(xs)
        features = preds.size(2)
        preds = preds.view(-1, features)
        ys = ys.view(-1)
        loss = self.loss(preds, ys)
        self.test_loss.update(loss)
        if self.extracting:
            raise ValueError("Extraction done, aborting")

    def test_epoch_end(self, outputs):
        self.log('test loss', self.test_loss)
        self.log('photon target', self.photon_target)

    def configure_optimizers(self):
        if self.use_adam:
            decay = set()
            no_decay = set()
            blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

            for mn, m in self.named_modules():
                for pn, p in m.named_parameters(recurse=False):
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if 'bias' in pn:
                        no_decay.add(fpn)
                    elif 'weight' in pn and not isinstance(m, blacklist_weight_modules):
                        decay.add(fpn)
                    else:
                        no_decay.add(fpn)

            param_dict = {pn: p for pn, p in self.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay

            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.02},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]

            optimizer = torch.optim.AdamW(optim_groups, lr=self.lr)
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2500, num_training_steps=self.training_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'name': 'Cosine LR scheduler'
                }
            }
        else:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=1e-5)
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2500, num_training_steps=self.training_steps)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'name': 'Cosine LR scheduler'
                }
            }
    
    def replace_output_head(self, module):
        self.transformer.output_head = module

    def enable_quantization(self):
        for m in self.transformer.modules():
            if isinstance(m, QuantizedLinear) or isinstance(m, QuantizedMatmul):
                m.enable_quantization()

    def set_photon_target(self, n_photons):
        self.photon_target = n_photons
        for m in self.transformer.modules():
            if isinstance(m, QuantizedLinear) or isinstance(m, QuantizedMatmul):
                m.set_photon_target(n_photons)

    def set_quantized_eval(self, value=True):
        for m in self.transformer.modules():
            if isinstance(m, QuantizedLinear) or isinstance(m, QuantizedMatmul):
                print("setting quantized eval")
                m.force_quantized_eval = value

    def save(self, fname):
        torch.save(self.transformer.state_dict(), fname)

    def load(self, fname):
        self.transformer.load_state_dict(torch.load(fname))

    def enable_extraction(self):
        lin1 = self.transformer.decoders[0].ff.layer2
        lin1.extract_simulated = True
        lin1.extract_name = 'first_linear'
        lin2 = self.transformer.decoders[-1].ff.layer2
        lin2.extract_simulated = True
        lin2.extract_name = 'last_linear'
        attn1 = self.transformer.decoders[0].attn.qmm1
        attn1.extract_simulated = True
        attn1.extract_name = 'first_attn'
        attn2 = self.transformer.decoders[-1].attn.qmm1
        attn2.extract_simulated = True
        attn2.extract_name = 'last_attn'
        self.extracting = True
    