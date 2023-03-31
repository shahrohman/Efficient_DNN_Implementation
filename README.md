# Efficient DNN Implementation on FPGA

The objective of this project is to optimize a pretrained deep neural network model by quantization/pruning/compression and implement it on a Xilinx PYNQ-Z2 board using the FINN Framework.

## Requirements

* Python >= 3.7 .
* Ubuntu 20.04 for FINN Compiler (other versions may work but are not tested).
* PYNQ-Z2 board with pynq v2.6 image (other versions may work but are not tested).
* Vivado 2022.1 (2020.1 or below for Vitis-HLS - Not working on 2021.1 and 2021.2 for some reason).
* GPU training-time acceleration (*Optional* but recommended).


## Training DNN

We want to  train our own quantized neural network (QNN) using the Brevitas library, which supports quantization-aware training (QAT). This will involve experimenting with various quantization levels to find out the most energy-efficient model. Once the QNN model has been trained, it must be exported to FINN-ONNX format, which is a format compatible with the FINN Framework.

### Prerequisites
Brevitas library can be installed by running the following command:

```bash
pip install brevitas
```

Check out available info at https://xilinx.github.io/brevitas/getting_started for getting started with Brevitas.

Check out brevitas/notebooks for examples on how to use Brevitas.

## Exporting to FINN-ONNX

The section below is mainly based on the FINN [getting started](https://finn.readthedocs.io/en/latest/getting_started.html) guide. All of this was tested on Ubuntu 20.04.

### Installing docker

Run the following commands to install docker:

```bash
chmod +x docker_install.sh
./docker_install.sh
```

You can test your docker installation by running the ```hello-world``` container:

```bash
docker run --rm hello-world
```

The ```--rm``` flag tells docker to remove the container once it's done. Your output should look like:

```bash
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

### Setting up the FINN docker container

Modify ```FINN_XILINX_PATH```, ```FINN_XILINX_VERSION``` and ```VIVADO_PATH``` environment variables pointing to your Xilinx installation path and Vivado installation path in ```env_finn.sh```.
Then run the following commands to set up the FINN docker container:

```bash
chmod +x path_finn.sh
./path_finn.sh
```

### Verifying the FINN installation

Clone the FINN compiler from the repository: ```git clone https://github.com/Xilinx/finn/``` and go into the directory: ```cd finn```.
Execute ```./run-docker.sh quicktest``` to verify your installation.

If everything is set up correctly, you can now run the following command to open jupyter notebook:

```bash
bash ./run-docker.sh notebook
```

This will launch the Jupyter notebook server inside a Docker container, and print a link on the terminal that you can open in your browser to run the FINN notebooks or create new ones. The link will look something like this (the token you get will be different): ```http://127.0.0.1:8888/?token=1dbc348eff2e275f9fdc9e61d9e9564fe5cf8e0d259a8642```. Open it in your browser and you are ready to go.

### Exporting the model to FINN-ONNX

Next step is to export the trained QNN model to FINN-ONNX format. This can be done by following this [tutorial](https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import.ipynb).


## Creating the dataflow graph

FINN’s build_dataflow system must be applied to the exported model to create a dataflow graph for the FPGA platform. This step involves mapping each layer of the QNN to a Vivado HLS description, parallelizing each layer’s implementation to the appropriate degree, and utilizing on-chip FIFOs to connect the layers and create the full accelerator.
To create the dataflow graph, you can follow this [tutorial](https://github.com/Xilinx/finn/blob/main/notebooks/end2end_example/cybersecurity/3-build-accelerator-with-finn.ipynb).

## Implementing the DNN on the FPGA

Implement the generated bitstream on the FPGA.
