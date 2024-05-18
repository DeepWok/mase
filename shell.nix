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
    # This executes some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    # TODO: consider use setuptoolsBuildHook, as documented in https://nixos.org/manual/nixpkgs/stable/#python
    pythonPackages.venvShellHook
    # pythonPackages.torch-bin

    # houskeeping 
    git
    neovim
    glib
  ];
  postShellHook = ''
    # install mase as a package
    python3 -m pip install .
  '';
}
