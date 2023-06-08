Loes Scoring using AlexNet
==========================

Loes Scoring
------------

The Loes score is an adrenoleukodystrophy (ALD) MRI score, which rates the severity of white matter lesions and ranges from 0 (normal) – 34 (abnormal) [12]. MRI's were scored by two independent physicians (IH and MS). The physicians were blinded to the neuropsychological test results.

---from [Overall intact cognitive function in male X-linked adrenoleukodystrophy adults with normal MRI](https://ojrd.biomedcentral.com/articles/10.1186/s13023-019-1184-4)
by Noortje J. M. L. Buermans, Sharon J. G. van den Bosch, Irene C. Huffnagel, Marjan E. Steenweg, Marc Engelen, Kim J. Oostrom & Gert J. Geurtsen

34 distinct parts of the brain are examined (through MRIs) for evidence of ALD.
Each section is given a score of 0 or 1 depending on the abscence or presence
of the disease in that region.

AlexNet
-------

AlexNet is the name of a convolutional neural network (CNN) architecture, designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton, who was Krizhevsky's Ph.D. advisor.

---from ["AlexNet"](https://en.wikipedia.org/wiki/AlexNet)

Method
------

We train an instance of AlexNet to predict the Loes score *in toto* rather than piece-by-piece.
