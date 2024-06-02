from flask import Flask, render_template, redirect, url_for, session, request, flash

app = Flask(__name__)
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'



@app.route("/messaging", methods=["GET", "POST"])
def messaging():
    if request.method == "GET":
        data = {
            "shovon": "some messages for shovon",
            "arsho": "other messages for arsho"
        }
        receiving_user = None
        messages = None
        if 'receiving_user' in session:
            receiving_user = session["receiving_user"]
        messages = data.get(receiving_user)
        return render_template("/home/agent/workspace/MediNoteAI/src/medinote/inference/messaging.html", messages=messages, clicked_user=receiving_user)

    if request.method == "POST":
        receiving_user = request.form.get("recipient")
        session["receiving_user"] = receiving_user
        flash("Sent!")
        return redirect(url_for("messaging"))
    
    
    
if __name__ == "__main__":
    app.run(debug=True,port=8888,host='0.0.0.0')