Gianluca Elia
Aalborg University Copenhagen
Sound and Music Computing
elgiano@gmail.com
gelia20@student.aau.dk
NO UiO's Studentweb

## Timbre interpolation in latent space

### Goals
The project aims at developing a system for timbre interpolation, possibly usable in real-time as well as for offline rendering.

### Background
The project is needed for a specific task: interpolating sounds from a pianist and a vocalist, to create a third sound that mixes their acoustics. Machine learning was chosen as a method to gain elements there were not conceived by the agency of sound designers, thus embracing artifacts and distortions, welcoming a non-personal third voice in the process.

### Design
The chosen architecture is a Variational Autoencoder. The idea is that training such a network on sounds from both voice and piano, blends can be obtained by interpolating their respective embeddings.
An architecture design that is claimed to work in real-time is published by [J. T. Colonel, S. Keene - Conditioning Autoencoder Latent Spaces for Real-Time Timbre Interpolation and Synthesis (2020)](
https://deepai.org/publication/conditioning-autoencoder-latent-spaces-for-real-time-timbre-interpolation-and-synthesis), with code available at [JITColonel/manne](https://github.com/JTColonel/manne).

### Process

#### Adapting the code
A substantial amount of work has been put in making the code from J.T.Colonel work, especially understanding it, adapting it to Tensorflow 2, and deploying a cleaned-up version for the present task. A function was added to interpolate between embeddings from two different files.

#### Selecting samples for a dataset
A dataset has to be put together with samples from both voice and piano. Starting from separate studio recordings of the two, fragments are selected and put together, reducing the amount of silences in between sounds. However, soft decays, especially from piano, are preserved, as they fit into the concept of the project, dealing with mixed acoustics.

#### Running the code
Instructions for pre-processing, training and rendering in `README.md`. Note that there are no examples or code for running the system in real-time, feature that will be explored in a subsequent work.

#### Notes on IFFT and phase
When interpolating from two sources, we are currently applying phase information from the first one, unchanged, and discarding it from the second. Other methods can be attempted, such as combining the phases or reconstructing them, for instance from the derivative of synthesized spectral magnitudes

### Evaluation
The architecture is evaluated by its original authors, as described in their paper. For our task, rendered interpolations will be evaluated by ear, for their suitability in the broader artistic project.
