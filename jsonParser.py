import json
import os
import heapq
hashtag_set = set()
new_hashtag_dictionary = dict()
# get scaled tweet evaluation
def get_tweet_evaluation(tweet_evaluation):
    if tweet_evaluation>0:
        return 1
    elif tweet_evaluation==0:
        return  0
    else:
        return -1
def count_positive_negative_neutral_top_hashtags():
    hashtag_evaluator_file = open('hashtag_evaluator.txt', 'r')
    positive_count = 0
    negative_count = 0
    for line in hashtag_evaluator_file.readlines():
        line_parts = line.split(',')
        evaluation = int(line_parts[2])
        if evaluation<0:
            negative_count+=1
        elif evaluation>0:
            positive_count+=1
    print(positive_count)
    print(negative_count)
    print(600 - (positive_count+negative_count))
def update_hashtag_evaluation():
    hashtag_evaluator_file = open('hashtag_evaluator.txt', 'r')
    current_hashtag_dictionary = dict()
    current_evaluation_dictionary = dict()
    for line in hashtag_evaluator_file.readlines():
        #print(line)
        line_parts = line.split(',')
        if len(parts)<3:
            print(line)
        hashtag = line_parts[0]
        count = int(line_parts[1])
        current_hashtag_dictionary[hashtag]=count
        current_evaluation_dictionary[hashtag]= int(line_parts[2])

    hashtag_evaluator_file.close()
    updated_hashtag_file = open('hashtag_evaluator.txt', 'w')
    k_hashtags_sorted = heapq.nlargest(450, new_hashtag_dictionary, key=new_hashtag_dictionary.get)
    sorted_current_hashtags = heapq.nlargest(len(current_hashtag_dictionary), current_hashtag_dictionary, key=current_hashtag_dictionary.get)
    for top_hashtag in k_hashtags_sorted:
        #print(top_hashtag + " : " + str(new_hashtag_dictionary[top_hashtag]))
        #if top_hashtag not in current_hashtag_dictionary:
        current_hashtag_dictionary[top_hashtag] = new_hashtag_dictionary[top_hashtag]
            #updated_hashtag_file.write(top_hashtag + ',' + str(new_hashtag_dictionary[top_hashtag]) + '\n')
    for hashtag in sorted_current_hashtags:
        if hashtag in current_evaluation_dictionary:
            updated_hashtag_file.write(hashtag + ','+  str(new_hashtag_dictionary[hashtag]) +','
                             + str(current_evaluation_dictionary[hashtag])   + '\n')
    for hashtag in current_hashtag_dictionary:
        if hashtag not in current_evaluation_dictionary:
            updated_hashtag_file.write(hashtag + ',' + str(new_hashtag_dictionary[hashtag])  + '\n')

        #else:

#my_dir = 'CoVaxxyDataset'
def get_evaluation_dictionary():
    hashtag_evaluator_file = open('hashtag_evaluator.txt', 'r')
    current_evaluation_dictionary = dict()
    for line in hashtag_evaluator_file.readlines():
        print(line)
        line_parts = line.split(',')
        hashtag = line_parts[0]
        count = int(line_parts[1])
        current_evaluation_dictionary[hashtag] = int(line_parts[2])
    return current_evaluation_dictionary


json_directory = 'covaxxy_output'
#output_directory = 'tweet_outputs'
no_hashtag_tweets_file = open('tweet_outputs/no_hashtags_tweets.txt','w', encoding="utf8")
labeled_tweet_output_file = open('tweet_outputs/labeled_tweets.txt','w', encoding="utf8")
labeled_tweet_output_file.write('id;separator;'+'text' + ';separator;' + 'sentiment' + '\n')
file_count = 0
hashtag_count = 0
total_tweet_count = 0
evaluation_dictionary = get_evaluation_dictionary()





for item in os.listdir(json_directory):
    file_count+=1
    if file_count<1:
        continue
    if os.path.isfile(os.path.join(json_directory, item)):
        json_file =  open(json_directory+ "/"+item,'r', encoding="utf8")
        for line in json_file.readlines():
            parts = line.split(';EndOfJson;')
            total_tweet_count+=1
            #print(parts[1])
            json_text_dictionary = json.loads(parts[0])
            #print(parts[1])
            if parts[1]!="#NoHashTag\n":
                #print('hi')
                hashtags = parts[1]
                # remove # sign
                hashtags = hashtags.replace('#','')
                hashtags = hashtags.lower()
                hashtags = hashtags.encode("ascii", "ignore").decode()
                #remove duplicate hashtags
                if hashtags.find(',')!=-1:
                    hashtags = hashtags.replace(',',' ')
                    hashtags = " ".join(sorted(set(hashtags.split()), key=hashtags.index))
                    #print(hashtags)
                tweet_evaluation = 0
                for hashtag in hashtags.split():
                    hashtag_set.add(hashtag)
                    if hashtag in new_hashtag_dictionary:
                        new_hashtag_dictionary[hashtag]+=1
                    else:
                        new_hashtag_dictionary[hashtag]=1
                    if hashtag in evaluation_dictionary:
                        tweet_evaluation+= evaluation_dictionary[hashtag]
                #write tweet with labels to output file
                tweet_evaluation = get_tweet_evaluation(tweet_evaluation)
                if 'text' in json_text_dictionary :
                    tweet_text = json_text_dictionary['text'].replace('\n', " ")

                else:
                    tweet_text = json_text_dictionary['full_text'].replace('\n', " ")
                tweet_id = str(json_text_dictionary['id'])
                labeled_tweet_output_file.write(tweet_id+';separator;'+tweet_text + ';separator;' + str(tweet_evaluation) + '\n')
            else:

                #print(json_text_dictionary)
                if  'text' in json_text_dictionary :
                    tweet_text = json_text_dictionary['text'].replace('\n', " ")
                    no_hashtag_tweets_file.write(tweet_text+'\n')
                else:
                    tweet_text = json_text_dictionary['full_text'].replace('\n', " ")
                    no_hashtag_tweets_file.write(tweet_text + '\n')
                hashtag_count += 1



count_positive_negative_neutral_top_hashtags()
print('Number of hashtags: '+str(len(hashtag_set)))
update_hashtag_evaluation()

statistics_output = open('output_stats.txt','w')

statistics_output.write('Total count: '+str(total_tweet_count)+'\n')
statistics_output.write('Hashtag count: '+ str(total_tweet_count- hashtag_count)+'\n')
# some JSON:
#x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
#y = json.loads(x)

# the result is a Python dictionary:
#print(y["age"])