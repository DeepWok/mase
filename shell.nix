with import <nixpkgs>{
  config.allowUnfree = true;  
};

let
  pythonPackages = pkgs.python3Packages;
  MACHOP_DIR = builtins.toString ./machop;
in pkgs.mkShell {

  LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.clangStdenv.cc.cc pkgs.libGL pkgs.glib cudaPackages.cuda_cudart cudaPackages.cudatoolkit cudaPackages.cudnn ];

  name = "impureMaseEnv";
  venvDir = "./.venv";
  buildInputs = [
    # A Python interpreter including the 'venv' module is required to bootstrap
    # the environment.
    pythonPackages.python

    # This executes some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    pythonPackages.venvShellHook

    # For MASE, in order to compile some binary extensions they may
    # require, the Python modules listed in the requirements.txt need
    # the following packages to be installed locally:
    taglib
    openssl
    git
    libxml2
    libxslt
    libzip
    zlib
    libGL
    glib
    libglvnd
    mesa

    python311Packages.cocotb-bus
    python311Packages.cocotb    
    python311Packages.pyopengl
    wget
    python311Packages.tensorboard
    # python311Packages.tensorflowWithCuda
    python311Packages.tensorflow-bin
    python311Packages.torch-bin
    python311Packages.torchvision-bin
    python311Packages.torchaudio-bin

    cudaPackages.cuda_cudart
    cudaPackages.cudatoolkit
    cudaPackages.cudnn

    # python311Packages.pytorch-lightning
    python311Packages.keras
    python311Packages.python-lsp-server

	verible
    verilator
    svls
  ];

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    TMPDIR=./tmp/ pip install -r ./machop/requirements.txt
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH

    # Add macho to the users PYTHONPATH
    export PYTHONPATH="$PYTHONPATH:${MACHOP_DIR}"

    export CUDA_PATH=${pkgs.cudatoolkit}
    # Enter terminal shell of your choice e.g. fish or bash
    # fish
  '';
}
