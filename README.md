# [Next Day Wildfire Spread(kaggle open dataset)](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread?datasetId=1726926)
Data analysis/modeling repo for the kaggle 'Next Day Wildfire Spread' open dataset.

This repo comforms to the [Udacity Git Commit Message Style Guide](https://udacity.github.io/git-styleguide/).
***
### [Workaround for TF2.11 optimizer op issue on Apple M2](https://developer.apple.com/forums/thread/721619)
> (post quot)
> Hi @ppobar
I assume you are seeing this on the latest wheels with tensorflow-macos==2.11 and tensorflow-metal==0.7.0? In that case this most probably has to do with recent changes on tensorflow side for version 2.11 where a new optimizer API has been implemented where a default JIT compilation flag is set (https://blog.tensorflow.org/2022/11/whats-new-in-tensorflow-211.html). This forces the optimizer op to take an XLA path that the pluggable architecture has not implemented yet causing the inelegant crash as it cannot fall back to supported operations. Currently the workaround is to use the older API for optimizers that was used up to TF 2.10 by exporting it from the .legacy folder of optimizers. So more concretely by using Adam optimizer as an example one should change
> ```python
> from tensorflow.keras.optimizers import Adam
> ```
> to
> ```python
> from tensorflow.keras.optimizers.legacy import Adam
> ```
> This should restore previous behavior while the XLA path support is being worked on. Let me know if this solves the issue for you! And if not, could you provide details on which OS version, tf-macos and tf-metal versions you are seeing this and a script I can use to reproduce the issue?
***
### The Type
The type is contained within the title and can be one of these types:

* feat: A new feature
* fix: A bug fix
* docs: Changes to documentation
* style: Formatting, missing semi colons, etc; no code change
* refactor: Refactoring production code
* test: Adding tests, refactoring test; no production code change
* chore: Updating build tasks, package manager configs, etc; no production code change
