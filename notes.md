# Notes

- Sampling:
    - am Anfang werden mit einer uniform policy steps gesampled
        - n_initial_exploration_steps: Wie viele Steps werden am Anfang mit uniform policy gemacht
        - wird gerundet auf vielfache von epoch_length
    - ab einem anderen Zeitpunkt wird angefangen zu trainieren (erst sollen genug samples im replay buffer sein)
        - min_replay_buffer_size: Wie viele Steps müssen im Replay buffer stehen, damit mit dem Training begonnen wird


# Logging
- Sollte Zeit (timestamps) und env steps enthalten um nach Zeit und Sampling Effizienz zu unterscheiden.
- Hyperparameter settings sollen am Anfang der csv auftauchen, muss beim Einlesen der Daten beachtet werden!
- Verschiedene runs werden in einer csv gespeichert und können über die Spalte 'seed' voneinander unterschieden werden.
- Logging von TD3 muss gecheckt und ggf. überarbeitet werden, damit wir alle Daten erhalten die wir plotten wollen.

# Fragen

- animationen in der präsentation? wie am besten?
- installation von mujoco erklären? welches OS?
- im paper werden ko-varianzen erwähnt, aber im code scheint es nur separate gauss-funktionen zu geben.
    - output = 2 * action_space