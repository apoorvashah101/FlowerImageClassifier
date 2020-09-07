
import argparse
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import json
import tensorflow as tf
import numpy as np




# Predict and Process images

def predict(image_path, model_path, top_K, category_path):

    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    predictions = model.predict(image)
    
    top_ps, top_k_indices = tf.math.top_k(predictions, top_k)
  
    prob = top_ps[0].numpy()
    
    label = top_k_indices[0].numpy()
    
    print(label)
  # print(prob)
    
    if category_path != None:

        cl_names = class_names(category_path)

        label_out = np.array([])
        print(label)

        for i in label:

            label_out = np.append(label_out, cl_names[str(i+1)])
            print(label)
    
        label = label_out
    return prob , label



def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, size = (224,224))
    image /= 255
    image = image.numpy()
    return image

def class_names(category_path):

    with open(category_path, 'r') as f:

        class_names = json.load(f)

    return class_names




parser = argparse.ArgumentParser(description='Given an image of a flower with shape (224,224,3), the program predicts the top K most likely flower classes')

parser.add_argument('img_p', help='Filepath to Image')
parser.add_argument('model_p', help='Filepath to Model .H5')
parser.add_argument('-k','--top_k', type=int, help='This returns the top K classes along with their probabilities', default = 1)
parser.add_argument('-c','--category_names', help='Filepath to class names JSON file')
args = parser.parse_args()


if __name__=='__main__':
   
    top_k = args.top_k

    
    

    catg = True

    if args.category_names == None:

        catg = False

    prob, class_name = predict(args.img_p, args.model_p, top_k, args.category_names)

#    print(class_name[0]+1)
    
    print('\n\n')
    print('{:20}'.format('Class Name') ,'{:20}'.format('Probability'))
    
    top_prob = np.argmax(prob,axis=0)
    
    for i in range(top_k):

        if catg == False:
         
            if (i == 0):
                class_name[top_prob] = class_name[top_prob] +1
            print('{:20}'.format(str(class_name[i])), '{:3f}'.format(prob[i]))                
        else:

            print('{:20}'.format(class_name[i]), '{:3f}'.format(prob[i]))
   

    print('\n\n* The image "', args.img_p, '" belongs to class: {:10}'.format(class_name[top_prob]), ', with a probability of {:3f}'.format(prob[top_prob]))
