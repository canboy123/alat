# Attack-less adversarial training for a robust adversarial defense
## Introduction
We have downloaded and extended from https://github.com/tensorflow/cleverhans.<br/>
Please check the original version from the link above.<br/>
<br/>
If you have found some missing script, please download the cleverhans github first and replace our scripts into the corresponding folders.<br/>
Please contact us if you still cannot solve the issue.<br/>

Step:
<ol>
  <li>Download all files from <a href="https://github.com/tensorflow/cleverhans">cleverhan</a> github.</li>
  <li>Replace our files into corresponding folder.</li>
  <li>
    Execute the correct python script in "<i>cleverhans_tutorials</i>" (or "<i>examples</i>") folder.<br/>
    E.g.: <br/>
    <i>cleverhans_tutorials/mnist_tutorial_tf_single_pr.py</i> for MNIST dataset<br/>
    <i>examples/ex_cifar10_tf_multiple_pr.py</i> for CIFAR10 dataset
  </li>
  <li>
    Go to correspoding <i>utils_tf.py</i> to set different cases for the experiments. The setting is located at <i>model_eval()</i> function with a variable name "case". Note that case=0 is a normal case, case=1 is used for Case A, case=2 is used for Case B, case=3 is used for Case C, and case=4 is used for Case D.<br/>
    E.g.: <br/>
    <i>cleverhans/utils_tf_pr_mnist.py</i> is used for <i>cleverhans_tutorials/mnist_tutorial_tf_single_pr.py</i>.<br/>
    <i>cleverhans/utils_tf_multiple_pr_cifar10.py</i> is used for <i>examples/ex_cifar10_tf_multiple_pr.py</i>
  </li>
</ol>

## Citing this work
```
@article{ho2022attack,
  title={Attack-less adversarial training for a robust adversarial defense},
  author={Ho, Jiacang and Lee, Byung-Gook and Kang, Dae-Ki},
  journal={Applied Intelligence},
  volume={52},
  number={4},
  pages={4364--4381},
  year={2022},
  publisher={Springer}
}
```
