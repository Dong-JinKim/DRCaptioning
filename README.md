# DRCaptioning
---
The instruction of the relational captioning.

# Preliminaries
Our model is implemented in [Torch](http://torch.ch/), and depends on the following packages: 
* [torch/torch7](https://github.com/torch/torch7)
* [torch/nn](https://github.com/torch/nn) 
* [torch/nngraph](https://github.com/torch/nngraph) 
* [torch/image](https://github.com/torch/image) 
* [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson)
* [qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd)
* [jcjohnson/torch-rnn](https://github.com/jcjohnson/torch-rnn)

After installing torch, you can install / update these dependencies by running the following:
```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec
```

# Test
Then point your web browser to [http://localhost:8181/view_results.html](http://localhost:8181/view_results.html).

If you have an entire directory of images on which you want to run the model, use the `-input_dir` flag instead:

```bash
th run_model.lua -input_dir /path/to/my/image/folder
```

# Results 







----
# Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
