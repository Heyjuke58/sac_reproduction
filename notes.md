# Sachen die anders/besonders sind

- Sampling:
    - am Anfang werden mit einer uniform policy steps gesampled
        - n_initial_exploration_steps: Wie viele Steps werden am Anfang mit uniform policy gemacht
        - wird gerundet auf vielfache von epoch_length
    - ab einem anderen Zeitpunkt wird angefangen zu trainieren (erst sollen genug samples im replay buffer sein)
        - min_replay_buffer_size: Wie viele Steps müssen im Replay buffer stehen, damit mit dem Training begonnen wird
- im paper werden ko-varianzen erwähnt, aber... TODO
    - output size = 2 * action_space
- im policy loss wird im code nur eins der beiden value functions für KL benutzt
- policy loss hat auch regularization:
- reihenfolge der gradient updates ist im code anders
- die log_sigmas werden im code in `[-20, 2]` geclippt -> sigmas in `[2.1e-9, 7.4]`

# Logging
- Sollte Zeit (timestamps) und env steps enthalten um nach Zeit und Sampling Effizienz zu unterscheiden.
- Hyperparameter settings sollen am Anfang der csv auftauchen, muss beim Einlesen der Daten beachtet werden!
- Verschiedene runs werden in einer csv gespeichert und können über die Spalte 'seed' voneinander unterschieden werden.

# Fragen

- animationen in der präsentation? wie am besten?
- installation von mujoco erklären? welches OS?

# TODO

- eval für sac (daten speichern)
- tests für sac
- plotten

## Checklist
