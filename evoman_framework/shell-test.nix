{ pkgs ? import <nixpkgs> {} }:

(pkgs.python3.withPackages (ps : with ps; [
  numpy
  pygame
])).env
