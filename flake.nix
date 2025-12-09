{
  inputs = {
    # Use nixos-25.05 to support uv package (consistent with Coprocessor_for_Llama)
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05"; 
    systems.url = "github:nix-systems/default-linux";
    
    # flake-utils helps with cross-platform support
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs: let
    # Helper function library
    lib = nixpkgs.lib;
  in
    # Generate outputs for default Linux system architecture
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
      };
    in rec {
      # ---------- Formatter ----------
      formatter = pkgs.alejandra;

      # ---------- Development Shells ----------
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # --- Python (from shell.nix) ---
          python311
          python311Packages.pip
          python311Packages.sphinx

          # --- General dev / utils (from shell.nix) ---
          wget
          just
          sphinx
          git
          neovim
          glib
          unzip
          mesa
          libglvnd  # OpenGL library (contains libGL.so.1 needed by opencv-python)
          cmake
          zsh

          # --- Hardware tools (from shell.nix) ---
          svls
        ]
        # Conditionally add verible on Linux (from shell.nix)
        ++ lib.optionals pkgs.stdenv.isLinux [
          pkgs.verible
        ];

        nativeBuildInputs = with pkgs; [
          uv
        ];

        # Shell Hook: Print version information and export library paths
        shellHook = ''
          echo ">>> Toolchain versions:"
          echo "CMake:        $(cmake --version | head -n1 2>/dev/null || echo not found)"
          echo "Python 3.11:  $(python3.11 --version 2>/dev/null || echo not found)"
        '';
      };
    });
}