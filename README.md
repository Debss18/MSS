# ITD-Aware Binaural Music Source Separation

This repository contains my extension to the Binaural-MSS framework for binaural music source separation. The main contribution is an interaural time difference (ITD)-aware loss integrated into an HDemucs-style training pipeline to better preserve spatial cues in separated binaural stems.

Overview

This project fine-tunes a small HDemucs-style source separation model on binaural MUSDB-style mixtures using a composite loss consisting of:

waveform reconstruction loss
ITD-based spatial loss using GCC-PHAT
four-stem separation: drums, bass, other, vocals

The goal is to improve spatial consistency and localization cues in separated binaural outputs.

Relationship to Existing Repositories

This project extends:

https://github.com/richa-namballa/binaural-mss and
https://github.com/facebookresearch/demucs

This repository does not redistribute code from Binaural-MSS or Demucs.

To run the scripts in this repository, first clone and set up the upstream Binaural-MSS repository and its dependencies.

What This Repository Adds

This repository contains only my extension code:

src/itd_loss.py
-GCC-PHAT-based ITD extraction and ITD loss
src/composite_loss.py
-Combined waveform + ITD-aware loss
scripts/train_itd_hdemucs_musdb.py
-Training modifications for ITD-aware HDemucs fine-tuning
example training, inference, and evaluation commands
