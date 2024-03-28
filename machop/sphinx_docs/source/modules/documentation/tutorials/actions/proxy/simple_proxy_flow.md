## Proxy Command

roxy command can be easily executed in MASE command line interface.



## Command line interface
Use the following command to generate a meta-proxy. Shown below is the command needed to generate meta-proxy with sample configuation.

```bash
cd machop
./ch proxy --config configs/nas/proxy_nas.toml
```
The generated meta-proxy weight is saved in path `nas_results/meta_proxy/meta_proxy.pt`.