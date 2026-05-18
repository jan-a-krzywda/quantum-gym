### Embedding the "Brain" into the Circuit: Low-Latency Shadow Tomography on Real Hardware




Highly constrained environments often drive the most radical leaps. The same holds true for near-term quantum hardware. 

#### TL;DR
Recently, driven by a requirement of quickly characterizing quantum state on a real hardware, I attempted to implement a shadow tomography protocol on the Quantum Inspire platform. However, I quickly faced a severe hardware bottleneck: the Quantum Inspire backend lacked support for parameterized circuits needed for shot-by-shot sampled Pauli measurements. I was forced to choose between running the exact same circuit for thousands of shots (yielding poor tomographic data) or running thousands of different circuits in separate jobs (suffering from crippling queue and network latencies). Than, I thought about the third way, I've designed a novel protocol for shadow tomography that bypasses the classical-quantum bottleneck entirely by embedding the classical decision process directly into the quantum circuit.

#### The Measure-First Paradigm
Finding classical representations of a quantum state in polynomial time has completely changed how we approach quantum information processing. Shadow-based algorithms have implemented a "measure first, ask questions later" paradigm, effectively treating the quantum computer as a feature processor that generates data for a classical machine learning model.

The most common realization of this paradigm is shadow tomography via random Pauli measurements. Theoretically, it is elegant and straightforward: apply a random single-qubit rotation, then read out in the computational basis. In practice, however, this requires a uniquely randomized circuit for every single shot. On many cloud-based platforms, this constant classical-to-quantum ping-pong introduces massive latency. 

To overcome this, we need a protocol compatible with any platform that supports non-demolition (QND) mid-circuit measurements.

#### The "Hardware Softmax" Protocol
The key idea is to use a mid-circuit measurement to collapse a set of ancilla qubits into a random basis choice, which is then used to control the measurement basis of the data qubits. This allows us to effectively implement a random measurement process natively on the chip, without waiting for a classical compiler.

The protocol operates in six steps:
1. **Prepare** the data qubits in the state of interest.
2. **Initialize** a set of ancilla qubits in a superposition state that encodes the desired probability distribution over the measurement bases.
3. **Measure** the ancillas mid-circuit, collapsing them into a specific basis choice.
4. **Feed-forward** the measurement outcome to control the application of single-qubit rotations on the data qubits, physically altering the measurement basis for that specific shot.
5. **Readout** the data qubits in the standard computational basis.
6. **Reconstruct** the classical shadow by feeding the collected data into a classical algorithm to estimate the quantum state.

#### The Hardware Gadget
The core of this protocol relies on a specific "gadget": one data qubit paired with two dedicated ancilla qubits. If our data qubits are kept in a line, this ammounts to a tri-linear connectivity pattern where each data qubit is directly connected to two ancillas. As a proof of principle, we concentrate on a 3-qubit cluster state, which requires 3 data qubits and 6 ancillas in total, the topology of which is shown below:

The two ancillas act as our classical dice. One ancilla controls the $X$ vs $Z$ basis choice via a controlled-$R_y(-\pi/2)$ rotation. The second ancilla controls the $Y$ vs. $Z$ basis choice via a controlled-$R_x(\pi/2)$ rotation.


If we initialize the ancillas in the $|+\rangle$ state, they yield a uniform distribution over the outcomes. Because the controlled rotations apply based on these outcomes, the data qubit's measurement basis maps out as follows:
* **`|00>`** $\rightarrow$ $Z$ basis (no rotation), 25% probability.
* **`|10>`** $\rightarrow$ $X$ basis (via $R_y(-\pi/2)$), 25% probability.
* **`|01>`** $\rightarrow$ $Y$ basis (via $R_x(\pi/2)$), 25% probability.
* **`|11>`** $\rightarrow$ $Y$ basis (via $R_y(-\pi/2)$ followed by $R_x(\pi/2)$), 25% probability.

Notice a slight collision? Due to the non-commuting nature of the rotations, the `|11>` outcome also results in a $Y$-basis measurement. This means a naive $|+\rangle$ initialization actually measures $Y$ 50% of the time! 

To account for this, we can easily correct the bias by adjusting the initial rotation angles of the ancillas. We map our desired target probabilities $P(X)$, $P(Y)$, and $P(Z)$ into precise physical rotation angles $\theta_1$ and $\theta_2$:

```python
def softmax_to_angles(px: float, py: float, pz: float) -> Tuple[float, float]:
    p2 = py 
    p1 = px / (px + pz + 1e-12)   

    theta1 = 2.0 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
    theta2 = 2.0 * np.arcsin(np.sqrt(np.clip(p2, 0, 1)))
    return theta1, theta2
```
We then apply these angles to the ancillas during the initialization step:
```python
    for i, dq in enumerate(DATA_QUBITS):
        a1, a2 = ANCILLA_MAP[dq]
        if theta1_per_qubit[i] != 0:
            qc.ry(theta1_per_qubit[i], a1)
        if theta2_per_qubit[i] != 0:
            qc.ry(theta2_per_qubit[i], a2)
    qc.barrier()
```
To get a uniform distribution px = py = pz = 1/3, we need theta1 = pi/4 and theta2 = 70.5 deg.

#### Results & Next Steps
To validate the protocol, I ran this shadow tomography procedure on quantum inspire to characterize a 3-qubit cluster state, which has three stabilisers K_i|psi> = c_i|psi>. As you can see in the density matrix heatmaps below, qualitatively the reconstructed density beautifully matches the ideal state, proving that the mid-circuit logic executes the randomized measurements.

However the reduced amplitudse of off-diagonal density matrix elements suggest presence of the imperfections, which are most likely dominated by the limited fidelity of gates and non-demolition measurements on the hardware.  

This is confirmed when we look at the fidelity between the targer state and the reconstructed one and the reconstructed stabilizer expectation values, which are all significantly reduced from the ideal value of 1. For comparsion, we also plot the observable which is expected to be zero. 

While this demonstration utilized a uniform distribution over the three Pauli bases, the true power of this "Hardware Logic" lies in its tunability. In, particulair I am planning to use the protocol to implement a more active learning-based shadow tomography, where the measurement distribution is adaptively updated based on the data collected in previous shots. While it cannot achive the same level of universality as the original classical shadow protocol, I believe it can be more powerful in the presence of noisy, biased and limited data, which is the reality of near-term quantum hardware.