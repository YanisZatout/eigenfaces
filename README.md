# eigenfaces
  A quick class I made to implement the eigenfaces algorithm with a reconstruction feature. Basically, after calculating the SVD of the database of faces, we can extract a basis for faces. This basis can then be used for a wide range of applications. My class allows you to give it any database of images and create this basis for its faces. Execute `code_test.py` if you want to have a quick test with the provided files. Don't hesitate to ask questions I'm always available !
  
# Some quick results:
On the training set:
![train_set](https://user-images.githubusercontent.com/72573031/115107316-5f59c300-9f6a-11eb-91ae-82477350d4f0.png)
On the testing set:
![test_set](https://user-images.githubusercontent.com/72573031/115107318-62ed4a00-9f6a-11eb-909d-61d4645cd750.png)

For this specific example, we took only 100 directions to recreate the eigenfaces, representing roughly 50 % of the variance of the faces.
