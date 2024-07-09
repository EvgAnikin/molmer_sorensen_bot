# Mølmer–Sørensen bot
A Telegram bot emulating experimental parameters adjustment for Molmer-Sorensen entangling gates in cold trapped ions

This Telegram bot simulates the Mølmer-Sørensen (MS) Hamiltonian in the simplest case of two ions interacting with a single phonon mode. and illuminated by a bichromatic laser beam:
$$H=\dots$$
Here $\omega$ is the frequency of the phonon mode; $\mu_+$ and $\mu_-$ are the detunings of the bichromatic laser beam components; $\Delta\omega$ is the qubit transition frequency shift, and $\Omega_\pm$ are the amplitudes of bichromatic beam components. T

To get an ideal MS gate, one should satisfy conditions
1. $t = 2\pi/|\delta|$, where $\delta=\mu-\omega$
2. $\Omega_+=\Omega_-\approx \delta/(2\eta)$
3. $\delta\omega=0$
When trying to satisfy these conditions in a real experiment, one encouters the problem that some system and laser parameters are not known precisely. That is, the qubit transition frequency is affected by magnetic field drifts; laser frequency itself and
phonon mode frequency are prone to drifts; the amplitudes of the bichromatic beam are determined by voltages applied to an AOM and cannot be set independently. Therefore, the controllable experimental parameters should be fine-tuned to match the unknown system parameters and satisfy conditions 1-3.

In our bot, we assume that the trapped-ion system is described by the parameters $\omega$, $\eta$ and $\delta\omega$. They have values $\omega \approx 2\pi \times 1\mathrm{MHz}$, $\eta \approx 0.1$, and 
$|\delta\omega| < 2\pi \times 10\mathrm{kHz}$, and their precise values are unknown to the experimentalist (the user). The experimentalist is able to control the following parameters:
