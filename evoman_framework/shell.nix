{ pkgs ? import <nixpkgs> {} }:
let
    mach-nix = import (
        builtins.fetchGit {
            url = "https://github.com/DavHau/mach-nix/";
            ref = "refs/tags/3.5.0";
        }
    ) {};
    custom-python = mach-nix.mkPython {
        requirements = ''
            leap_ec
            numpy
            pygame
        '';
    };
in
with pkgs;
mkShell {
  buildInputs = [
    custom-python
  ];
}
