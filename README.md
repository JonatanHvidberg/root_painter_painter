## RootPainter

Described in the paper "RootPainter: Deep Learning Segmentation of Biological Images with Corrective Annotation" (RootPainterPaper)

For original sorse:
https://github.com/Abe404/root_painter


# Evaluation of CNN Using CNN

## Abstract from report

In image analysis with pixel-wise segmentation U-Net is often used. Here the corrective annotation strategy can be used.
Corrective Annotation Strategy and Active Learning both use human-in-the-loop. Therefore it will make sense to use them together. We have made a correction-model which evaluates how good a segmentation from a segmentation-model is, by guessing how many pixels are being corrected. This is done by using U-Net with the image and segmentation as input.

However, this gives slightly ambiguous results, as it does not give much better statistical results than using the uncertainty as an indicator for how much is corrected.

The correction model finds large areas which it believes should be corrected. This would probably make it easier for an expert to correct. It therefore does not give a clear answer to how good the method could be to use for active learning.


## The idea

Corrective Annotation Strategy Described in RootPainterPaper there is an expert there Corrective a model
Using the output of the model and the picture, can a nother model (correction-model) be trained to predict what the expert would correct?

## The code

There's no need to reinvent the wheel. So there is just made some changes RootPainterPaper U-net so it takes 4 channels instead of 3.
where the 4. channel is the output from the first model.

## Some of the data

The model is in is in this case predicting biopors

Turquoise: model prediction
Green: expert correct to background
Reb: expert correct to foreground

![MarineGEO circle logo](/MarkdownIMG/B13-1_003_sa.jpg "expert and model")

By taking the entropy of the softmax of the model can be used as uncertainty

This goes from a scale blue is 100% certan to green is 0% certan

![MarineGEO circle logo](/MarkdownIMG/B13-1_003_u.jpg "uncertainty")

softmax of the correction-model is the percent prediction that the expert correct it

![MarineGEO circle logo](/MarkdownIMG/B13-1_003_p.jpg "softmax of the correction-model")

