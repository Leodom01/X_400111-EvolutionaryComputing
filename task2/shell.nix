{ pkgs ? import <nixpkgs> {} }:
let
  commit = "73de0e330fd611769c192e98c11afc2d846d822b";  # from: Mon Apr 27 2020
  fetchPypi = import (builtins.fetchTarball {
    name = "nix-pypi-fetcher";
    url = "https://github.com/DavHau/nix-pypi-fetcher/tarball/${commit}";
    # Hash obtained using `nix-prefetch-url --unpack <url>`
    sha256 = "1c06574aznhkzvricgy5xbkyfs33kpln7fb41h8ijhib60nharnp";
  });
in
(pkgs.python3.withPackages (ps : with ps; [
  numpy
  pygame
  matplotlib
  pandas
  scipy
  cma
  deap
#  numba
  pymoo
  scipy
  (fetchPypi "comocma" "0.5.1")
])).env
