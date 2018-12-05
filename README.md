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

To run the model on new images, use the script `run_model.lua`. To run the model on a test image,
use the following command:

```bash
th run_model.lua -input_image /path/to/my/image/file -output_vis_dir /path/to/the/output/folder
```


If you have an entire directory of images on which you want to run the model, use the `-input_dir` flag instead:

```bash
th run_model.lua -input_dir /path/to/my/image/folder -output_vis_dir /path/to/the/output/folder
```


# Results 
The resulting output file format is as follows:

```
{
{
	"boxes": [
		[9.4456, 46.8276,569.0354, 368.3203],
		[183.6740, 77.7138, 185.4196, 332.1285],
		[403.1037, 77.593994, 323.3377, 334.4553],
		...
		]
	"captions": [
  'the man wearing black shirt',
  'the man has head',
  'the man wearing a white shirt',
  ...
  ]
  
}
...
}
```





----
# Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
