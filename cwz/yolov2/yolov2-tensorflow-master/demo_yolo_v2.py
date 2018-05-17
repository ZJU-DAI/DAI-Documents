from utils.im_transform import imcv2_recolor, imcv2_affine_trans
from utils import box
import time
import os
import numpy as np
import tensorflow as tf
import cv2
from utils import tool 
import json

#from multiprocessing.pool import ThreadPool
#pool = ThreadPool()
#slim = tf.contrib.slim
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]='3'

class YOLO_detector(object):
    def __init__(self):
        model_name = 'yolov2-coco'
        model_dir = './models'
        self.gpu_utility = 0.9
        self.pb_file = '{}/{}.pb'.format(model_dir, model_name)
        self.meta_file = '{}/{}.meta'.format(model_dir, model_name)
        self.batch = 4
        self.graph = tf.Graph()
        with tf.device('/gpu:1'):
            with self.graph.as_default() as g:
                self.build_from_pb()
                gpu_options = tf.GPUOptions(allow_growth=True)
                sess_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
                self.sess = tf.Session(config = sess_config)
                self.sess.run(tf.global_variables_initializer())
        return
    
    def build_from_pb(self):
        with tf.gfile.FastGFile(self.pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # 将graph_def保存的图加入到当前默认的图
        tf.import_graph_def(graph_def, name="")
        
        with open(self.meta_file, "r") as fp:
            self.meta = json.load(fp)
        #Placeholders
        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')

#        self.setup_meta_ops()
        
    def setup_meta_ops(self):
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
            })
        utility = min(self.gpu_utility, 1.0)
        if utility > 0.0:
            print('GPU model with {} usage'.format(utility))
            cfg['gpu_options'] = tf.GPUOptions(per_process_gpu_memory_fraction = utility)        
            cfg['allow_soft_placement'] = True
        else:
            print('Run totally on CPU')
            cfg['device_count'] = {'GPU': 0}

        self.sess = tf.Session(config = tf.ConfigProto(**cfg))
        self.sess.run(tf.global_variables_initializer())

    def resize_input(self, im):
        h, w, c = self.meta['inp_size']
        imsz = cv2.resize(im, (w, h))
        imsz = imsz / 255.
        imsz = imsz[:,:,::-1]
        return imsz
    
    def process_box(self, b, h, w, threshold):
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = self.meta['labels'][max_indx]
        if max_prob > threshold:
        	left  = int ((b.x - b.w/2.) * w)
        	right = int ((b.x + b.w/2.) * w)
        	top   = int ((b.y - b.h/2.) * h)
        	bot   = int ((b.y + b.h/2.) * h)
        	if left  < 0    :  left = 0
        	if right > w - 1: right = w - 1
        	if top   < 0    :   top = 0
        	if bot   > h - 1:   bot = h - 1
        	mess = '{}'.format(label)
        	return (left, right, top, bot, mess, max_indx, max_prob)
        return None
       
    def preprocess(self, im, allobj = None):
        """
        """
        if type(im) is not np.ndarray:
        	im = cv2.imread(im)
        
        if allobj is not None: # in training mode
        	result = imcv2_affine_trans(im)
        	im, dims, trans_param = result
        	scale, offs, flip = trans_param
        	for obj in allobj:
        		_fix(obj, dims, scale, offs)
        		if not flip: continue
        		obj_1_ =  obj[1]
        		obj[1] = dims[0] - obj[3]
        		obj[3] = dims[0] - obj_1_
        	im = imcv2_recolor(im)
        
        im = self.resize_input(im)
        if allobj is None: return im
        return im#, np.array(im) # for unit testing
    
    def postprocess(self, net_out):
        meta = self.meta
        result = box.box_constructor(meta,net_out)
        return result

    def detect_object(self, im):
        this_inp = self.preprocess(im)
        expanded = np.expand_dims(this_inp, 0)
        inp_feed = list()
        feed_dict = {self.inp: expanded}
        inp_feed.append(expanded)
        feed_dict = {self.inp : expanded}    
       
        print("Forwarding the image input.")
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        
        time_value = time.time()
        last = time_value - start
        print('Cost time of run = {}s.'.format(last))
        result = self.postprocess(out[0])
        last = time.time() - time_value
        
        print('Cost time of postprocess = {}s.'.format(last))
        return result

    #新添加
    def close_session(self):
        self.sess.close()
    #结束
        
def demo_image():
    yolo = YOLO_detector()
    colors = yolo.meta['colors']
    img_dir = "./images"
    image_names = tool.find_files(img_dir) 
    for filename in image_names:
        im = cv2.imread(filename)
        h,w,_ = im.shape
        results = yolo.detect_object(im) 
        thick = int((h + w) // 300)
        draw = im.copy()
        h, w, _ = draw.shape
        for i in range(len(results)):
#            cv2.putText(draw,str(results[i]['category']),(int(w*results[i]['x1']),int(h*results[i]['y1'])-12), 0, 1e-3*h, colors[results[i]['label']], thick//3)
            cv2.putText(draw,"{},{:.3f}".format(str(results[i]['category']), results[i]['score']),(int(w*results[i]['x1']),int(h*results[i]['y1'])-12), 0, 1e-3*h, colors[results[i]['label']], thick//4)
            cv2.rectangle(draw,(int(w*results[i]['x1']),int(h*results[i]['y1'])),(int(w*results[i]['x2']),int(h*results[i]['y2'])), colors[results[i]['label']], thick)
        cv2.imshow("result", draw)
        cv2.waitKey()
      
def demo_video():
    yolo = YOLO_detector()
    colors = yolo.meta['colors']
    video_name = 'test1.mp4'
    data_dir = "./videos" 
    video_file = os.path.join(data_dir, video_name)
    
    print(video_file)
    vcap = cv2.VideoCapture(video_file)

    width, height, FPS = int(vcap.get(3)), int(vcap.get(4)), vcap.get(5)
    video_out = cv2.VideoWriter()
    output_path = './videos/demo1.mp4'
    video_out.open(output_path,  # Filename
                   cv2.VideoWriter_fourcc(*'DIVX'),
                   # Negative 1 denotes manual codec selection. You can make this automatic by defining the "fourcc codec" with "cv2.VideoWriter_fourcc"
                   FPS,  # 10 frames per second is chosen as a demo, 30FPS and 60FPS is more typical for a YouTube video
                   (width, height),  # The width and height come from the stats of image1
                   )

    if False == vcap.isOpened():
        print("video cannot open!\n")
        return -1
    idx = 0
    while True:
        idx += 1
        ret, img = vcap.read()
        if False == ret:
            break
        print('video is read')
        im = img
        h,w,_ = im.shape
        start = time.time()
        array = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        results = yolo.detect_object(array) 
        last = (time.time() - start)
        thick = int((h + w) // 300)
        draw = array.copy()
        h, w, _ = draw.shape
        for i in range(len(results)):
            cv2.putText(draw,"fps:{}".format(1/last),(1,18), 0, 1e-3*h, colors[results[i]['label']], thick//3)
            cv2.putText(draw,"{},{:.3f}".format(str(results[i]['category']), results[i]['score']),(int(w*results[i]['x1']),int(h*results[i]['y1'])-12), 0, 1e-3*h, colors[results[i]['label']], thick//3)
            cv2.rectangle(draw,(int(w*results[i]['x1']),int(h*results[i]['y1'])),(int(w*results[i]['x2']),int(h*results[i]['y2'])), colors[results[i]['label']], thick)
        
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", draw)
#        cv2.waitKey()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #image.save(os.path.join(output_path, image_file), quality=90)
        video_out.write(cv2.cvtColor(np.array(draw), cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    print("Done")

if __name__ == '__main__':
    print("run demo...")
    demo_video()
#    demo_image()

