let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-unstable";
  pkgs = import nixpkgs { config = {allowUnfree = true;}; overlays = []; };
in

let
  pythonPackages = pkgs.python311Packages;
in pkgs.mkShellNoCC {
  venvDir = "./.venv";
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
  packages = with pkgs; [
    # Python 3.11
    pythonPackages.python
    pythonPackages.pip
    pythonPackages.sphinx
    # This executes some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    # TODO: consider use setuptoolsBuildHook, as documented in https://nixos.org/manual/nixpkgs/stable/#python
    pythonPackages.venvShellHook

    # houskeeping 
    wget
    just
    sphinx
    git
    neovim
    glib
    unzip
    mesa
    cmake
    zsh

    # hardware
    # verible is only supported on Linux (x86_64-linux, i686-linux and aarch64-linux)
    # https://search.nixos.org/packages?channel=23.11&show=verible&from=0&size=50&sort=relevance&type=packages&query=verible
    svls
  ]
  ++ (pkgs.lib.optionals pkgs.stdenv.isLinux [ verible ]);
  postShellHook = ''
    # install mase as a package
    sudo -H python3 -m pip install -e .
    # add env variables 
    source scripts/init-nix.sh
  '';
}
