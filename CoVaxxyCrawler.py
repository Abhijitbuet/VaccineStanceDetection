import os
import tweepy
from tweepy import OAuthHandler
import json
import time
consumer_key = 'L0sbLoXOYZwTTf5LaMbS42s3W'
consumer_secret = 'b4kMzjma9Qvv8H44lKlMLAdNpXDNTlIX5DFlmGcuCEt7nQW4wJ'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth)



my_dir = 'CoVaxxyDataset'
output_dir = 'covaxxy_output'
file_count = 0
for item in os.listdir(my_dir):
    file_count+=1
    print(file_count)
    if file_count<180:
        continue
    if os.path.isfile(os.path.join(my_dir, item)):
        file = open(my_dir+ "/"+item,'r')
        output_file = open(output_dir + "/" + item + '.json_output.txt', 'a+', encoding="utf8")
        tweetCount = 0
        print(file)
        for line in file.readlines():
            #time.sleep(1)
            #print(line)
            try:
                tweet = api.get_status(line, tweet_mode = 'extended')
            except (tweepy.errors.NotFound,tweepy.errors.Forbidden) as e:
                continue
            except tweepy.errors.TwitterServerError as ex:
                time.sleep(300)
                continue
            except tweepy.errors.HTTPException as httperror:
                print('http exception')
                time.sleep(60)
                continue
            full_text = ''
            try:
                full_text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                full_text = tweet.full_text
            #print(full_text)
            number_of_hashtags = len(tweet.entities['hashtags'])
            json_str = json.dumps(tweet._json)
            #print(json_str)
            hashtags = ''
            if number_of_hashtags==0:
                hashtags = '#NoHashTag'
            else:
                i = 0

                for hashtag in tweet.entities['hashtags']:
                    if i == 0:
                        hashtags += hashtag['text']
                    else:
                        hashtags = hashtags +','+ hashtag['text']
                    i+=1


            #break
            output_file.write(json_str+';EndOfJson;'+hashtags+'\n')
            tweetCount+=1
            if tweetCount>3000:
                break
            print(item+" "+str(tweetCount))
            time.sleep(0.8)

