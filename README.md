## Host Multiple TensorFlow computer vision models using Amazon SageMaker Multi-model endpoint

Amazon SageMaker Multi-Model Endpoints provides a scalable and cost-effective way to deploy large numbers of custom machine learning models. SageMaker Multi-Model endpoints will let you deploy multiple ML models on a single endpoint and serve them using a single serving container. Your application simply needs to include an API call with the target model to this endpoint to achieve low latency, high throughput inference. Instead of paying for a separate endpoint for every single model, you can host many models for the price of a single endpoint. For detailed information about multi-model endpoints, see Save on inference costs by using Amazon SageMaker multi-model endpoints.

In this repository, we demonstrate how to host two computer vision models trained using the TensorFlow framework under one SageMaker multi-model endpoint. For a detailed full walkthrough of the example covered in this repo, see this accompanying [AWS blog post](https://aws.amazon.com/blogs/machine-learning/host-multiple-tensorflow-computer-vision-models-using-amazon-sagemaker-multi-model-endpoints/). For the first model, we train a smaller version of AlexNet CNN to classify images from the CIFAR-10 dataset. For the second model, we use a pretrained VGG16 CNN model pretrained on the ImageNet dataset and fine-tuned on the Sign Language Digits Dataset to classify hand symbol images. 

<img src="/img/sagemaker-design-patterns-mme-cv.jpg" alt="SageMaker Multi-model Endpoint"/>

### Model-1: CIFAR-10 image classification
For model-1, we will use the CIFAR-10 dataset. CIFAR-10 is a benchmark dataset for image classification in the CV and ML literature. CIFAR images are colored (three channels) with dramatic variation in how the objects appear. It consists of 32×32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. 

### Model-2: Sign language digits classification 
For model-2, we will use the sign language digits dataset. This dataset distinguishes the sign language digits from 0 to 9. The figure below shows a sample of the dataset. 
Following are the details of the dataset: 

* Number of classes = 10 (digits 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9) 
* Image size = 100 × 100
* Color space = RGB
* 1,712 images in the training set 
* 300 images in the validation set
* 50 images in the test set 


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

