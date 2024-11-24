from Decision_Tree_Model import predict_new_data, train_model
from Feature import write_feature_file
from traindata_get import trans_data
def Q1():
    #将分类好的图片转为可训练数据
    # trans_data(base_path = 'Q1_train_data')
    # train_model('train_data.csv', 5)
    # print('train_model done')

    #将新图片转为特征文件并预测
    image_folder='Attachment'
    feature_csv='features.csv'
    predicted_classes_csv='predicted_classe.csv'
    max_images=4399
    write_feature_file(image_folder,feature_csv,max_images)
    predict_new_data(feature_csv, image_folder,predicted_classes_csv)
    print('predict_new_data done')
    #将新图片转为特征文件并预测
    image_folder='Attachment2'
    feature_csv='features2.csv'
    predicted_classes_csv='predicted_classe2.csv'
    max_images=4399
    write_feature_file(image_folder,feature_csv,max_images)
    predict_new_data(feature_csv, image_folder,predicted_classes_csv)
    print('predict_new_data done')

if __name__ == '__main__':
    Q1()
    