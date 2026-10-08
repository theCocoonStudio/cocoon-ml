# 02 Softmax cross-entropy

A layer outputs n raw scores. Softmax turns them into n positive numbers summing to one, by exponentiating each and dividing by the total. The loss is the negative log of the probability assigned to the correct class.

The gradient of that loss with respect to the raw scores is what backward needs. The known result is unusually simple; the task is to derive it and say why it is that simple.

Check it two ways: against finite differences, and against the same loss composed from the primitive operations, whose backward is the chain rule with nothing simplified.
