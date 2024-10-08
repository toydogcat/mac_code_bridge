from flask import Flask, session, redirect
from flask_sso import SSO

app = Flask('myapp')
ext = SSO(app=app)

#: Default attribute map
SSO_ATTRIBUTE_MAP = {
    'ADFS_AUTHLEVEL': (False, 'authlevel'),
    'ADFS_GROUP': (True, 'group'),
    'ADFS_LOGIN': (True, 'nickname'),
    'ADFS_ROLE': (False, 'role'),
    'ADFS_EMAIL': (True, 'email'),
    'ADFS_IDENTITYCLASS': (False, 'external'),
    'HTTP_SHIB_AUTHENTICATION_METHOD': (False, 'authmethod'),
}

app.config['SSO_ATTRIBUTE_MAP'] = SSO_ATTRIBUTE_MAP

@SSO.login_handler
def login_callback(user_info):
    """Store information in session."""
    session['user'] = user_info
    
@app.route('/')
def index():
    """Display user information or force login."""
    if 'user' in session:
        return 'Welcome {name}'.format(name=session['user']['nickname'])
    return redirect(app.config['SSO_LOGIN_URL'])

