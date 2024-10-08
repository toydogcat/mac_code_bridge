from flask import Flask
from flask_sso import SSO

app = Flask(__name__)
app.config['SSO_ATTRIBUTE_MAP'] = {
    'uid': 'uid',
    'email': 'email',
    'first_name': 'givenName',
    'last_name': 'sn'
}
app.config['SSO_LOGIN_URL'] = '/login'
app.config['SSO_LOGOUT_URL'] = '/logout'
sso = SSO(app)

@app.route('/')
def home():
    if 'sso_user' in app.config:
        user = app.config['sso_user']
        return f'Hello, {user["first_name"]} {user["last_name"]}! Your email is {user["email"]}.'
    return 'You are not logged in.'

@app.route('/login')
def login():
    return 'Login page'

@app.route('/logout')
def logout():
    return 'Logout page'

if __name__ == '__main__':
    app.run(debug=True)
