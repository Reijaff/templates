{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    gomod2nix.url = "github:nix-community/gomod2nix";
    gomod2nix.inputs.nixpkgs.follows = "nixpkgs";
    gomod2nix.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    gomod2nix,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      buildGoApplication = gomod2nix.legacyPackages.${system}.buildGoApplication;
      mkGoEnv = gomod2nix.legacyPackages.${system}.mkGoEnv;
    in {
      packages.default = buildGoApplication {
        pname = "myapp";
        version = "0.1";
        pwd = ./.;
        src = ./.;
        modules = ./gomod2nix.toml;
      };

      devShells.default = pkgs.mkShell {
        buildInputs = [
          (mkGoEnv {pwd = ./.;})
          gomod2nix.legacyPackages.${system}.gomod2nix

          pkgs.hello
        ];

        shellHook = ''
          $EDITOR ./
        '';
      };
    });
}
