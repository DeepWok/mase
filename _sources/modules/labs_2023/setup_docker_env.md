# ADLS Docker Environment Setup

This guide provides a step-by-step walkthrough for running the MASE ADLS Docker image and connecting VSCode to the ADLS container to execute labs. Using Docker containers ensures a uniform development environment across various platforms.

## Prerequisites

[VSCode](https://code.visualstudio.com/) and [Docker](https://www.docker.com/) need to be installed on your machine.

- Refer to [this page](https://docs.docker.com/desktop/) to install Docker on your Windows/Linux/Mac machine.
- Note that for Windows users, [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) is required.

## Steps

1. Setup X11 in case later you may run some GUI applications in the container like GTKWave to view waveforms. You need authenticate local applications to access the display server (the host).

    ```bash
    # on your host machine
    sudo apt install x11-xserver-utils
    xhost +local:
    ```

    For macOS users, the command mentioned above will not function properly. Instead, you can enable X11 forwarding by using xQuartz as an alternative solution.

    ```bash
    # on your host machine
    # if brew install fails, you will need to install homebrew
    brew install --cask xquartz
    open -a XQuartz
    xhost +localhost
    ```

2. You need a Docker Hub account to pull ADLS image.
    - Register for a free account on [Docker Hub](https://hub.docker.com/).
    - Login to Docker Hub from your [Docker Desktop](https://www.docker.com/products/docker-desktop/) or terminal (`docker login`).

3. Pull the [ADLS docker image](https://hub.docker.com/r/chengzhang98/mase-adls).

    ```bash
    # on your host machine
    docker pull chengzhang98/mase-adls
    ```

4. Run an ADLS container and allocate a pseudo-TTY connected to it.

    ```bash
    # on your host
    docker run --rm --name mase-adls \
    --net host -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY \
    -it chengzhang98/mase-adls:latest
    ```
    For macOS user that use xQuartz for x11, you will need to set up the DISPLAY env variable.
   
    ```bash
    # on your host
    docker run --rm --name mase-adls \
    --net host -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=docker.for.mac.host.internal:0\
    -it chengzhang98/mase-adls:latest
    ```


    Then you will be in the container's shell like the following:

    ```bash
    [20:36:05 root@docker-desktop / ] $
    📦
    ```

    You may enter `gtkwave` to test if the GUI application works. If it works, you will see a GTKWave window pop up.

    ```bash
    [20:36:05 root@docker-desktop / ] $
    📦 gtkwave # on your container
    ```

    You can enter `exit` to exit the container, but keep it running for now.

    > [!NOTE]
    > If you are interested in the details of the `docker run` command, please refer to [this page](https://docs.docker.com/engine/reference/commandline/run/).

5. Attach VSCode to the running container.

    - Open VSCode and install [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) to the VSCode on your host machine.
    - If you are using WSL2, you need to press `F1` to open the command palette, then type and select `WSL: Connect to WSL`. If you are using Linux/Mac, you can skip this step.
    - Press `F1` to open the command palette, then type and select `Dev Containers: Attach to Running Container...`. Then select the container named `mase-adls` you just started. VSCode will open a new window and attach to the container. You can see the container's shell in the VSCode's terminal.

6. Close VSCode and stop the container.

    - Close the VSCode window attached to the container.
    - In the container terminal in Step 4, enter `exit` to stop the container.

## Mount a local directory to the container

It's important to note that each time you exit and restart the container, any progress made within the container will be lost. To prevent this, you can mount a local directory to the container and store your work there, ensuring that your changes are preserved after you exit the container. Or you can choose to sync your work to a Github repository.

For example, you can mount the `./mase-tools` directory on your host machine to the `/workspace/mase-tools` directory in the container. Then attach VSCode to the container path `/workspace/mase-tools`.

```bash
# Similar to Step 4
# on your host machine
docker run --rm --name mase-adls \
    --net host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ./mase-tools:/workspace/mase-tools:z \
    -e DISPLAY \
    -it chengzhang98/mase-adls:latest
```
