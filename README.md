# OverNet_to_ONNX
In this repository, we show how to export OverNet described in [["OverNet: Lightweight Multi-Scale Super-Resolution with Overscaling Network"](https://openaccess.thecvf.com/content/WACV2021/papers/Behjati_OverNet_Lightweight_Multi-Scale_Super-Resolution_With_Overscaling_Network_WACV_2021_paper.pdf)] (for increasing spatial resolution within your network for tasks such as super-resolution) from PYTORCH to ONNX and running it using ONNX RUNTIME. 

### Requirements
- Python 3.5
- [PyTorch](https://github.com/pytorch/pytorch) (0.4.0), [torchvision](https://github.com/pytorch/vision)
- ONNX <https://github.com/onnx/onnx>
- ONNX Runtime <https://github.com/microsoft/onnxruntime>

### Steps
First, we need to train the model; however, we already trained OverNet for different scales including x2, x3, and x4. The pre-trained models are in the Checkpoints directory. 

1- To convert OverNet to ONNX:
```
# Scale x2
python OverNet_to_ONNX.py --ckpt_path './checkpoints/OverNet_test_x2.pth.tar' --image '/Images/x2/image_01_LR.png

# Scale x3
python OverNet_to_ONNX.py --ckpt_path './checkpoints/OverNet_test_x3.pth.tar' --image '/Images/x3/image_01_LR.png

# Scale x4
python OverNet_to_ONNX.py --ckpt_path './checkpoints/OverNet_test_x4.pth.tar' --image '/Images/x4/image_01_LR.png
```
2- In order to print a human readable representation of the graph and at the same time check if the model is well formed:
```
python ONNX_validate.py 
```
3- In order to run the model with ONNX Runtime:
```
#scale 2
python ONNXRUNTIME.py --output_dir 'outputs' --test_data_dir 'Images/x2'

#scale 3
python ONNXRUNTIME.py --output_dir 'outputs' --test_data_dir 'Images/x3'

#scale 4
python ONNXRUNTIME.py --output_dir 'outputs' --test_data_dir 'Images/x4'
```
