# Mølmer–Sørensen bot
A Telegram bot emulating experimental parameters adjustment for Molmer-Sorensen entangling gates in cold trapped ions

This Telegram bot simulates the Mølmer-Sørensen (MS) Hamiltonian in the simplest case of two ions interacting with a single phonon mode. and illuminated by a bichromatic laser beam:
$$H=\dots$$
Here $\omega$ is the frequency of the phonon mode; $2\mu$ is the detuning between two components of the bichromatic laser beam; $\Delta\omega$ is the qubit transition frequency shift, and $\Omega_\pm$ are the amplitudes of bichromatic beam components. T

To get an ideal MS gate, one should satisfy conditions
1. $t = 2\pi/|\delta|$, where $\delta=\mu-\omega$
2. $\Omega_+=\Omega_-\approx \delta/(2\eta)$
2. $\delta\omega=0$

