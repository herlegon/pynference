# Pynference
A tool to perform fast inference (PyTorch, TensorRT).<br/>
Nvidia GPU only.<br/>
**Proprietary Software, Closed License**<br/>


> [!IMPORTANT]
> With a consumer grade Nvidia GPU, for video or batch of images, performing a PyTorch or TensorRT inference with this tool is **fast**, even **[faster than some other tools coded in compiled language](./profiling.md)**. Explanations [here](./profiling.md).



# How to use
> [!IMPORTANT]
> Unless you have received a copy, you can't use this tool as it is currently not publicly released.

Though, some modules and libraries are released as open source: [pynnlib]()

## Supported hardware/software
- Windows 11
- Limited support on Linux: tested CPU inference only
- Nvidia GPU card
- Python 3.12
- Nvidia GPU driver: 552.22 or later


## Installation
- **Cuda 12.4 toolkit**
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11


> [!TIP]
> It is recommended to use the [miniconda distribution](https://docs.anaconda.com/free/miniconda/index.html) to create a separate python environment from the system python.

- **Create a conda environment**
    ```
    conda create -n pynference python==3.12
    conda activate pynference
    ```

- **Dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r .\requirements.txt
    ```
    ```bash
    pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
    ```

- (optionnal) Upgrade an already created environnment
    ```bash
    pip install -r .\requirements.txt --upgrade`
    ```

- **External tools**
```sh
python install.py
```


## Basic usage: examples

- Perform the **inference with a PyTorch model**, using the CUDA device 0 (default) and fp16 datatype
    ```bash
    python pynference.py --input in_video.mkv --output out_video.mkv --model model.pth --ep pytorch
    ```

- Perform the inference **using the TensorRT library**, using the CUDA device 0 (default) and fp16 datatype
    ```bash
    python pynference.py --input in_video.mkv --output out_video.mkv --model model.pth --ep trt
    ```

- Perform an inference and **copy audio track** to the output media file
    ```bash
    python pynference.py --input in_video.mkv --output out_video.mkv --model model.pth --ep trt --copy_audio
    ```

> [!CAUTION]
> The audio stream is copied only if no seek positon arguments is used.

- Perform an inference, **resize to 2K, use the SAR for the resize operation**, copy audio. The SAR value won't be set in the output video stream.
    ```bash
    python pynference.py --input in_video.mkv --output out_video.mkv --model model.pth --ep trt --final_resize_to 2K --final_resize_with_sar --copy_audio
    ```

- **Deinterlace** the input video with customized parameters, perform an inference.
    ```bash
    python pynference.py --input in_video.mkv --output out_video.mkv --model model.pth --ep trt --deint nnedi --deint_params nsize=s8x6:nns=n128:qual=slow:etype=s:pscrn=new3
    ```

## Arguments

### Input, Output and model

| Argument  | Description                   |
| :--- | :--- |
| `--input` | path to the input video file  |
| `--output`| path to the output video file |
| `--model` | path to the model file. Accepted extensions: `.pth`, `.pt`, `.onnx`, `.engine` |


### Execution provider

| Argument&nbsp; | Options   |  Default  | Description           |
| :--- | :---: | :---: | :--- |
| `--ep`       | `pytorch`, `trt` | `pytorch` | Execution Provider    |
| `--device` | `cuda`, `cpu`    | `cuda`    | Device used to perform the inference |
| `--fp`       | `fp16`, `fp32`   | `fp16`    | Datatype for the inference and model conversion. If not supported by the device, fallback to `fp32` |


### Options for the model conversion (PyTorch to TensorRT)
Refer to the documenation of [trtexec](https://docs.nvidia.com/tao/tao-toolkit/text/trtexec_integration/index.html)

| Argument&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Format    | Description   |
| :--- | :---: | :---|
| `--opset`     | int   | ONNX opset version, default: `17`    |
| `--min_size`  | `WxH`  | Mininum input video resolution. e.g. `640x480`  |
| `--opt_size`  | `WxH` or `input`  | Optimal input video resolution. Uses the dimension of the input video when set to `input` |
| `--max_size`  | `WxH`   | Maximum input video resolution  |
| `--fixed_size`|       | use the opt_size to set the min_size and max_size |
| `--opt_level` | `1` to `5`   | Optimization level, default: `3`|


### Seek position
Refer to [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html#toc-Main-options)

| Argument  | Format        |  Default  | Description           |
| :--- | :---: | :---: | :--- |
| `-ss`     | hh:mm:ss.ms   |           | seek start            |
| `-t`      | hh:mm:ss.ms   |           | duration              |
| `-to`     | hh:mm:ss.ms   |           | position              |


### Deinterlace

| Argument&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Options/format       |  Default  | Description           |
| --- | --- | --- | --- |
| `--deint`         | `nnedi`, `bob`, `bwdif`, `decomb`, `estdif`, `kerneldeint`, `mcdeint`, `w3fdif`, `yadif` | none | Algorithm used to deinterlace the input video  |
| `--deint_params` | str | `yadif` | Arguments passed to the filter. Refer to the [FFmpeg documentation](https://ffmpeg.org/ffmpeg-filters.html) |


### Resize


#### Input

| Argument&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Options/format| Description           |
| :---| :---: | :--- |
| `--discard_sar`           |               | Remove the SAR from the video stream without any size modification. |
| `--resize_with_sar`       |               | Use the SAR value to resize before the inference. SAR is removed from the video stream. |
| `--initial_scale`         | float         | Scale before the inference. If present, the SAR is copied to the output stream |
| `--initial_resize`        | `WxH`         | Resize before the inference. If present, the SAR is copied to the output stream |
| `--initial_resize_algo`   |  `bilinear`, `lanczos`, `bicubic`, `fast_bilinear`, `area`, `neighbor`, `bicublin`, `gauss`, `sinc`, `spline` | Algorithm used for the resize/scale operations. Default: `bicubic` when upscaling, `lanczos` when downscaling |

Notes:
- `--initial_scale` and `--initial_resize` are mutually exclusive. `--initial_scale` has priority.
- `--initial_resize` changes the display aspect ratio if not carefully choosen.


#### Output

| Argument&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Options/format|  Description           |
| --- | --- | --- |
| `--final_resize`      | `WxH`   | Resize the video to this dimension. The aspect ratio is conserved and borders are added if needed |
| `--final_resize_to`   |  `480p`, `720p`, `1080p`, `2K`, `1440p`, `2160p`,`4K`  | Make it easier to choose the output resizing parameter |
| `--final_resize_with_sar` |    | If used, the output video is resized accordingly with the SAR. Has no effect when `--final_resize` is used.  |
| `--final_resize_algo` |  `bilinear`, `lanczos`, `bicubic`, `fast_bilinear`, `area`, `neighbor`, `bicublin`, `gauss`, `sinc`, `spline` | Algorithm used for the resize operations. Default: `bicubic` when upscaling, `lanczos` when downscaling |

Notes:
- Any resize operation discards the SAR.
- `--final_resize` and `--final_resize_to` are mutually exclusive. `--final_resize` has priority.
- `--final_resize` changes the display aspect ratio if not carefully choosen.


#### Video encoding
Refer to the [FFmpeg documentation](https://ffmpeg.org/ffmpeg-all.html) |

| Argument&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Format        |  Default  | Description           |
| :--- | :---: | :---: | :--- |
| `--encoder`     |  `h264`, `h265`, `ffv1`, `vp9`  | `h264`          |                |
| `--pix_fmt`     |  rgb or yuv formats  |  `yuv420p`         |   recommended: `yuv420p`, `yuv420p10le`, `yuv420p12le`, `rgb24`, `rgb48`               |
| ~~`--colorspace`~~     | `none`, `to_bt709`, `as_bt709`   |  `none` | (no implemented yet) ~~Convert to bt709 before the inference, tag the output video as bt709 or do nothing~~               |
| `--color_range`     | `pc`, `tv`   |            |                |
| `--preset`     | `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow`   |           |                |
| `--crf`     |  0 to 51  |           | Constant Rate Factor (FFmpeg default: 23)               |
| `--tune`     |  `film`, `animation`, `grain`, `stillimage`, `fastdecode`, `zerolatency`  |           |                |


#### Audio

| Argument&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Description           |
| :--- | :--- |
| `--copy_audio`  | Copy the input audio stream to the output media. Warning: No copy will be done if one of the seek position or duration argument is used. Subtitles streams are also copied (experimental)|


## Not yet supported
- Inference with an ONNX model
- Conversion of the colorspace to BT.709 (Rec. 709)
- Customized input and output FFMpeg video filters
- Customized FFMpeg output video arguments


## Won't support
- NCNN
- AMD GPU
- Debug other containers than `.mkv`
- Transcode/copy audio streams when any seek position argument is used. Use another tool to merge audio and video streams: MKVToolNix, FFmpeg, ...

