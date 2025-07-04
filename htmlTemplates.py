css = '''
<style>
body {
    background-color: #1e1e1e;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}

/* Fixed input container */
.input-container {


    width: 60%;
    background-color: #2b313e;
    padding: 0rem;
    border-radius: 5px;

}

/* Text input */
.input-container input[type="text"] {
    flex: 1;
    padding: 0.7rem;
    border-radius: 5px;
    border: none;
    background-color: #1e1e1e;
    color: white;
}

/* Send button */
.input-container button {
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 0.7rem 1rem;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 1rem;
    border-radius: 5px;
    cursor: pointer;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
