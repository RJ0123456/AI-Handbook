In the mathematical framework of LSTM, $h_t$ is called the hidden state. If $C_t$ is the LSTM's "internal ledger," then $h_t$ is its final "work report" presented to the outside world.

We can understand it from the following three dimensions:

## 1. It is the result of "deliberation"

Review the formula: $h_t = o_t * \tanh(C_t)$

* $\tanh(C_t)$: First, the complete "internal memory" (cell state) at this moment is compressed into the range $(-1, 1)$, extracting the key information.

* $o_t$ (output gate): This is a filter. It determines which parts of the current memory are useful for predictions at the current moment.

* Multiplication: Only information "approved" by the output gate can ultimately be displayed through $h_t$.

## 2. Its Two Roles

$h_t$ plays a dual role in the model, acting as a bridge between the preceding and following steps:

* External Output: It is the direct output of the current time step (for example, in a translation task, it might correspond to the predicted next word).

* Internal Transmission: It serves as reference information, passing it to the next time step ($t+1$). It tells the "gate" of the next time step: "I just output this content; use this to decide what to remember and what to forget next."

## 3. A Figurative Analogy

We can continue with the example of "reading a novel":

* $C_t$ (cell state): This is all the plot details stored in your mind after reading five chapters (who died, who took the knife, who lied).

* $h_t$ (latent state): This is the summary you give when a friend asks you, "What was this chapter about?" after you finish reading the chapter.

* You don't memorize every detail ($C_t$) in your mind.

* Based on the current problem (output gate $o_t$), you will extract the most important information from your mind: "The butler has been arrested."

## Summary

* $C_t$ is "long-term memory": it stores all the key information from the past, and is relatively primitive and comprehensive.

* $h_t$ is "short-term output": it is a transient expression of long-term memory in the current context.
