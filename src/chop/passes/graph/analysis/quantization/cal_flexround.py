# After quantize_transform_pass:


def calibrate_flexround(model, calibration_data, num_iters=1000):
    model.eval()
    optimizer = torch.optim.Adam([
        p for n, p in model.named_parameters() 
        if "log_s1" in n or "log_S2" in n or "log_s3" in n  # Only train quantization params
    ])
    
    for _ in range(num_iters):
        batch = calibration_data.next()  # Use a small calibration dataset
        output_quant = model(batch["input_values"])
        with torch.no_grad():
            output_fp = original_float_model(batch["input_values"])  # Original FP model
        
        loss = F.mse_loss(output_quant["last_hidden_state"], output_fp["last_hidden_state"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

