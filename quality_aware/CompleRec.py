import traceback
from os.path import join

import array
import tensorflow as tf
import pandas as pd
import gzip
import struct
import json
from pprint import pprint
import numpy as np
import pickle
import re, math
from collections import Counter
import json
import random
from itertools import permutations, islice
import argparse
import scipy.sparse

from contextlib import closing
import shelve
from operator import itemgetter 
from quality_aware import Encore
from sklearn.model_selection import KFold


def parase_args():
    parser = argparse.ArgumentParser(description="Run Encore.")
    parser.add_argument('--ImageDim', type=int, default=4096,
                        help='Image Dimension.')
    parser.add_argument('--TexDim', type=int, default=100,
                        help='Text Dimension.')
    parser.add_argument('--ImageEmbDim', type=int, default=10,
                        help='Image Embedding Dimension.')
    parser.add_argument('--TexEmDim', type=int, default=10,
                        help='Text Embedding Dimension.')
    parser.add_argument('--HidDim', type=int, default=100,
                        help='Hidden Dimension.')
    parser.add_argument('--FinalDim', type=int, default=10,
                        help='Fainl Dimension.')
    parser.add_argument('--learningrate', type=float, default=0.0001,
                        help='Learning Rate.')
    parser.add_argument('--trainchoice', nargs='?', default="Yes",
                        help='Training or Testing.')

    return parser.parse_args()



def main():
    args = parase_args()
    ImageDim, TexDim, ImageEmbDim, TexEmDim, HidDim, FinalDim, learningrate, trainchoice = args.ImageDim, args.TexDim, args.ImageEmbDim, args.TexEmDim, args.HidDim, args.FinalDim, args.learningrate, args.trainchoice



    with tf.device('/gpu:0'):
        encore = Encore.EncoreCell(ImageDim, TexDim, ImageEmbDim, TexEmDim, HidDim, FinalDim)

        encoreloss, thresh = encore.train()
        testacc = encore.predict_ratings()


        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
        #optimizer = tf.train.AdamOptimizer(learning_rate = learningrate)
        #print("AdadeltaOptimizer")
        train_op = optimizer.minimize(encoreloss)


        # Start training
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)


        #import data
        #AlsoBoughtRelationDic: {"bought_together":{item: [list of also bought items/bought together items]}}
        #AlsoBoughtInfoDic{item:[textual word count with smallercase, textual word count, text vector]}

        #my implement
        root = join('F://', 'Amazon Dataset', 'Electronics')
        def write_error(str, init=False):
            if init:
                with open(join(root, 'error.txt'),'w') as f:
                    f.write('error list\n')
            else:
                with open(join(root, 'error.txt'),'a') as f:
                    f.write(str)
        write_error("",True)
        '''
            asin_set:具有图像,文本,评分数据的商品ID集合 {asin}
            AlsoBoughtRelationDic:也购买商品关系 {}
        '''

        def read_asin_set():
            with open(join(root, 'available_asin_set.pickle'), 'rb') as f:
                asin_list=pickle.load(f)
                return asin_list
        asin_set=read_asin_set()

        def read_also_bought():
            with open(join(root, 'item_relation.pickle'), 'rb') as f:
                also_bought_dic=pickle.load(f)
                also_info={}
                for item in also_bought_dic['bought_together']:
                    also_info[item]=0
                return also_bought_dic,also_info
        AlsoBoughtRelationDic,AlsoBoughtInfoDic=read_also_bought()
        all_keys=AlsoBoughtRelationDic['bought_together'].keys()


        def read_rating():
            with open(join(root, 'item_rating.pickle'), 'rb') as f:
                return pickle.load(f)
        dscore=read_rating()
        #SubTraingKeys = random.sample(list(all_keys), len(all_keys)//4*3)
        SubTraingKeys = random.sample(list(all_keys), 11000)
        Testkeys = list(all_keys - set(SubTraingKeys))

        def read_text_vec():
            with open(join(root, 'doc_vec_Electronics.pickle'), 'rb') as f:
                return pickle.load(f)
        TextVecDic = read_text_vec()

        def read_visual(asin_set):
            def readImageFeatures(path):
                f = open(path, 'rb')
                while True:
                    asin = f.read(10)
                    if asin == '': break
                    a = array.array('f')
                    a.fromfile(f, 4096)
                    if asin not in asin_set:
                        continue
                    feature=np.asarray(a.tolist())
                    feature=feature.reshape((feature.shape[0],1))
                    yield {'asin': asin, 'feature':feature }

            img_path = join('F://', 'FDMDownload', 'image_features_Electronics.b')
            it = readImageFeatures(img_path)
            #debug
            #it=[{'asin': asin.encode('utf-8'), 'feature':np.zeros((1,4096)) }for asin in asin_set]
            imagefeature = {}
            try:
                for item in it:
                    asin = item['asin'].decode('utf-8')
                    if asin in asin_set:
                        imagefeature[asin] = item['feature']
            except EOFError as e:
                pass
            return imagefeature
        imagefeature=read_visual(asin_set)
        '''
        # replace with my implement
        with open('BoughtTogetherDic.pickle') as f:
            AlsoBoughtRelationDic,AlsoBoughtInfoDic = pickle.load(f)

        #TextVecDic: {item: [learned word vector]}
        with open('TextVecDic.pickle') as f:
            TextVecDic = pickle.load(f)

        #learned Bayesian score
        with open('RatingScore.pickle') as f:
            dscore = pickle.load(f)
        
        #image feature
        imagefeature = shelve.open('ImageBoughtTogetherDic.shelf')
        
        #train, test data
        with open("datakeys.pickle") as f:
            SubTraingKeys, Testkeys = pickle.load(f)
        '''


        #sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        if trainchoice == "Yes":  #five cross validation to train data
            print("train")
            kf = KFold(n_splits=5)
            
            result_acc = []

            for train_index, vali_index in kf.split(SubTraingKeys): #cross validation
                Trainkeys, Valikeys = np.array(SubTraingKeys)[train_index], np.array(SubTraingKeys)[vali_index]

                runtimes = 10

                for _ in range(runtimes):  #training times
                    random.shuffle(Trainkeys)
                    for key in Trainkeys:
                        #print(key)
                        try:
                            mi = np.transpose(imagefeature[key])
                            ti = TextVecDic[key].reshape((1,TexDim))
                            #print("try")
                        except Exception as e:
                            write_error('pos: 1 //'+traceback.format_exc())
                            traceback.print_exc()
                            continue
                        
                        #ti = np.transpose(TextVecDic[key])
                        checklist = AlsoBoughtRelationDic['bought_together'][key]
                        ichecklist = 0
                        
                        for it in checklist:
                            print("test checklist is", it)
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                print("begin")
                                #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:1})
                                '''_, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi,
                                                                                        encore.xj:mj, 
                                                                                        encore.tti:ti, 
                                                                                        encore.ttj:tj, 
                                                                                        encore.rscore:score, 
                                                                                        encore.y:1})'''
                                train_op.run(session=sess,feed_dict={encore.xi:mi,
                                                                                encore.xj:mj,
                                                                                encore.tti:ti,
                                                                                encore.ttj:tj,
                                                                                encore.rscore:score,
                                                                                encore.y:1})
                                #print('distance1 = %f', distance)
                                ichecklist = ichecklist + 1

                            except Exception as e:
                                write_error('pos: 2 //'+traceback.format_exc())
                                traceback.print_exc()
                                continue

                        flaglen = len(checklist)
                        NotRelationQ = set(all_keys) - set(checklist)  #it can be changed !!!!!!!!!!!!!!!!!!!!!!!

                        Qlist = list(NotRelationQ)
                        Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                        Qchecklist = list(itemgetter(*Qindex)(Qlist))
                        
                        ical = 0
                        
                        for it in Qchecklist:
                            if ical == ichecklist:
                                break
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:0})
                                '''_, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi, 
                                                                encore.xj: mj, 
                                                                encore.tti: ti, 
                                                                encore.ttj:tj, 
                                                                encore.rscore: score, 
                                                                encore.y:0})'''
                                train_op.run(session=sess, feed_dict={encore.xi: mi,
                                                                      encore.xj: mj,
                                                                      encore.tti: ti,
                                                                      encore.ttj: tj,
                                                                      encore.rscore: score,
                                                                      encore.y: 0})
                                #print('distance2 = %f', distance)
                                ical = ical + 1
                                #print("run2")
                            except Exception as e:
                                write_error('pos: 3 //'+traceback.format_exc())
                                traceback.print_exc()
                                continue


                

                print("begin test")
                for key in Valikeys[:2]:
                        print(key)
                        try:
                           mi = np.transpose(imagefeature[key])
                           ti = TextVecDic[key].reshape((1,TexDim))
                        except:
                            write_error('pos: 4 //'+traceback.format_exc())
                            traceback.print_exc()
                            continue
                        #ti = TextVecDic[key]
                        checklist = AlsoBoughtRelationDic['bought_together'][key]
                        ichecklist = 0

                        for it in checklist:
                            print("checklist is ", it)
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                                                      encore.xj: mj, 
                                                                                      encore.tti: ti, 
                                                                                      encore.ttj:tj, 
                                                                                      encore.rscore: score, 
                                                                                      encore.y:1})

                                print("test1 thresh is", test_thresh )
                                result_acc.append(cal_acc)
                                ichecklist = ichecklist + 1
                                #print("runtest")
                            except:
                                write_error('pos: 5 //'+traceback.format_exc())
                                traceback.print_exc()
                                continue

                        flaglen = len(checklist)

                        NotRelationQ = set(all_keys) - set(checklist)

                        Qlist = list(NotRelationQ)

                        Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                        Qchecklist = list(itemgetter(*Qindex)(Qlist))
                        
                        ical = 0

                        for it in Qchecklist:
                            if ical == ichecklist:
                                break
                            try:
                                mj = np.transpose(imagefeature[it])
                                tj = TextVecDic[it].reshape((1,TexDim))
                                score = np.array(dscore[it]).reshape([1,1])
                                cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                              encore.xj: mj, 
                                                              encore.tti: ti, 
                                                              encore.ttj:tj, 
                                                              encore.rscore: score, 
                                                              encore.y:0})


                                print("test2 thresh is", test_thresh )
                                result_acc.append(cal_acc)
                                ical = ical + 1
                                #print("runText2")
                            except:
                                write_error('pos: 6 //'+traceback.format_exc())
                                traceback.print_exc()
                                continue


            print("TexDim is %d" % TexDim)
            print("acc is")
            print(np.mean(result_acc))
            print("learning rate is")
            print(learningrate)
            print("runtime is %d"%runtimes)


        
        else:
            runtimes = 10
            for _ in range(runtimes):  #training times
                random.shuffle(SubTraingKeys)
                for key in SubTraingKeys[:10]:
                    print(key)
                    try:
                        mi = np.transpose(imagefeature[key])
                        ti = TextVecDic[key].reshape((1,TexDim))
                        #print("try")
                    except:
                        write_error('pos: 7 //' + traceback.format_exc())
                        traceback.print_exc()
                        continue
                    
                    #ti = np.transpose(TextVecDic[key])
                    checklist = AlsoBoughtRelationDic['bought_together'][key]
                    ichecklist = 0
                    
                    for it in checklist:
                        print("test checklist is", it)
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            print("begin")
                            #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:1})
                            _, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi, 
                                                                                    encore.xj: mj, 
                                                                                    encore.tti: ti, 
                                                                                    encore.ttj:tj, 
                                                                                    encore.rscore: score, 
                                                                                    encore.y:1})
                            print('distance1 = %f', distance)
                            ichecklist = ichecklist + 1
                            #tf.Print(sigma, [sigma], message="sigma is:")
                            #print("run")
                        except:
                            write_error('pos: 8 //' + traceback.format_exc())
                            traceback.print_exc()
                            continue

                    flaglen = len(checklist)

                    NotRelationQ = set(all_keys) - set(checklist)  #it can be changed !!!!!!!!!!!!!!!!!!!!!!!

                    Qlist = list(NotRelationQ)

                    Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                    Qchecklist = list(itemgetter(*Qindex)(Qlist))
                    
                    ical = 0
                    
                    for it in Qchecklist[:2]:
                        if ical == ichecklist:
                            break
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            #sess.run(train, feed_dict={xi:mi, xj: mj, tti: ti, ttj:tj, y:0})
                            _, distance = sess.run([encoreloss, thresh], feed_dict={encore.xi:mi, 
                                                            encore.xj: mj, 
                                                            encore.tti: ti, 
                                                            encore.ttj:tj, 
                                                            encore.rscore: score, 
                                                            encore.y:0})

                            print('distance2 = %f', distance)
                            ical = ical + 1
                            #print("run2")
                        except:
                            write_error('pos: 9 //' + traceback.format_exc())
                            traceback.print_exc()
                            continue


            result_acc = []

            print("begin test")
            for key in Testkeys:
                    print(key)
                    try:
                       mi = np.transpose(imagefeature[key])
                       ti = TextVecDic[key].reshape((1,TexDim))
                    except:
                        write_error('pos: 10 //' + traceback.format_exc())
                        traceback.print_exc()
                        continue
                    #ti = TextVecDic[key]
                    checklist = AlsoBoughtRelationDic['bought_together'][key]
                    ichecklist = 0

                    for it in checklist:
                        print("checklist is ", it)
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                                                  encore.xj: mj, 
                                                                                  encore.tti: ti, 
                                                                                  encore.ttj:tj, 
                                                                                  encore.rscore: score, 
                                                                                  encore.y:1})

                            print("test1 thresh is", test_thresh )
                            result_acc.append(cal_acc)
                            ichecklist = ichecklist + 1
                            #print("runtest")
                        except:
                            write_error('pos: 11 //' + traceback.format_exc())
                            traceback.print_exc()
                            continue

                    flaglen = len(checklist)

                    NotRelationQ = set(all_keys) - set(checklist)

                    Qlist = list(NotRelationQ)

                    Qindex = random.sample(range(1, len(NotRelationQ)), ichecklist + 100) 
                    Qchecklist = list(itemgetter(*Qindex)(Qlist))
                    
                    ical = 0

                    for it in Qchecklist:
                        if ical == ichecklist:
                            break
                        try:
                            mj = np.transpose(imagefeature[it])
                            tj = TextVecDic[it].reshape((1,TexDim))
                            score = np.array(dscore[it]).reshape([1,1])
                            cal_acc, test_thresh = sess.run([testacc, thresh], feed_dict={encore.xi:mi, 
                                                          encore.xj: mj, 
                                                          encore.tti: ti, 
                                                          encore.ttj:tj, 
                                                          encore.rscore: score, 
                                                          encore.y:0})


                            print("test2 thresh is", test_thresh )
                            result_acc.append(cal_acc)
                            ical = ical + 1
                            #print("runText2")
                        except:
                            write_error('pos: 12 //' + traceback.format_exc())
                            traceback.print_exc()
                            continue


            print("TexDim is %d" % TexDim)
            print("acc is")
            print(np.mean(result_acc))
            print("learning rate is")
            print(learningrate)
            print("runtime is %d"%runtimes)



        with open('EncoreParameter.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([sess.run(encore.Em), sess.run(encore.Et), sess.run(encore.W), sess.run(encore.E), sess.run(encore.b1), sess.run(encore.c)], f)



if __name__ == "__main__":
    main()
