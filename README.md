# deep-q-learning

![animation](./assets/animation.gif)

Minimal and Simple Deep Q Learning Implemenation in Keras and Gym

The code itself is 78 lines and is self documenting.


The explanation for the code is covered in the blog article [https://keon.io/rl/deep-q-learning-with-keras-and-gym/](https://keon.io/rl/deep-q-learning-with-keras-and-gym/)

I made minor tweaks to this repository.

I made the `memory` a deque instead of just a list. This is in order to limit the maximum number of the memory so we can give more weights to the more recent memories.

