let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-unstable";
  pkgs = import nixpkgs { config = {}; overlays = []; };
in

let
  pythonPackages = pkgs.python311Packages;
in pkgs.mkShellNoCC {
  venvDir = "./.venv";
  packages = with pkgs; [
    # Python 3.11
    pythonPackages.python
    pythonPackages.pip
    # This executes some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    # TODO: consider use setuptoolsBuildHook, as documented in https://nixos.org/manual/nixpkgs/stable/#python
    pythonPackages.venvShellHook

    # houskeeping 
    git
    neovim
    glib
  ];
}