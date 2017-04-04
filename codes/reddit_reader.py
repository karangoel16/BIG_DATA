import praw
import keys
import json
import time

def login():
    return praw.Reddit(client_id=keys.client_id,
                      client_secret=keys.client_secret,
                      user_agent=keys.user_agent)

def reddit_user_comment(reddit, user_name, count=None, comment_id=None):
    if count == None or comment_id == None:
        comments = reddit.get("/user/{}".format(user_name)).children
    else:
        comments = reddit.get("/user/{}?count={}&after=t1_{}".format(user_name, count, comment_id))
    return comments

def get_comments(redit, output_file, reddit_user="thedukeofetown"):
    with open(output_file, "w") as op_file:
        comments_count = 0
        previous_comment_id = None
        total_comments = 0
        comment_list = []
        while True:
            if total_comments == 0:
                comments = reddit_user_comment(redit, reddit_user)
            else:
                comments = reddit_user_comment(redit, reddit_user,
                                               count=total_comments,
                                               comment_id=previous_comment_id)
            comments_count = len(comments)
            for comm in comments:
                try:
                    data_dict = {
                                    "title": comm.link_title,
                                    "comment": comm.body,
                                    "created": comm.created,
                                    "comment_id": comm.id
                                }
                    comment_list.append(data_dict)
                except AttributeError:
                    print "error occured at count:", total_comments, " after=t1_{}".format(previous_comment_id)
                    continue
            previous_comment_id = comments[-1].id
            total_comments += comments_count
            print "total_comments received:", total_comments, previous_comment_id
            if comments_count < 25:
                break
            time.sleep(0.25)
        print len(comment_list)
        json.dump(comment_list, op_file, indent=4)

def main():
    reddit = login()
    print "successful login"
    output_file = "reddit_conversation.json"
    get_comments(reddit, output_file)
    print "finished"

if __name__ == '__main__':
    main()

