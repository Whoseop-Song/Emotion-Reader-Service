let eCount = [0,0,0,0,0,0,0];
let prevEmotion = 6;
function countEmotion(emotion) {
    // eCount[emotion] += 1;
    for (let j=0; j<emotion.length; j++){
        for (let i = 0; i < eCount.length; i++) {
            eCount[i] += emotion[j][i];
        }
    }
}

function postData(face) {
    let xhr = new XMLHttpRequest();
    xhr.open("POST", "/face", true);
    xhr.setRequestHeader("Content-type", "application/json");
    xhr.withCredentials = true;
    xhr.onreadystatechange = function () {
        if (xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200) {
            let emotion = JSON.parse(this.response);
            countEmotion(emotion);
        }
    };
    xhr.send(JSON.stringify(face));
}

function grab_face(e) {
    let canvas = document.getElementById("canvas");
    let context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    face_data = []
    count = 0
    e.data.forEach(rect => {
        context.drawImage(video, rect.x, rect.y, rect.width, rect.height, 0, 0, 48, 48);
        data = context.getImageData(0, 0, 48, 48);
        face_data[count] = [];
        for (let i = 0; i < data.data.length; i += 4) {
            face_data[count].push(0.299 * data.data[i] + 0.587 * data.data[i + 1] + 0.114 * data.data[i + 2]);
        }
        count += 1;
    });
    if (count > 0) {
        postData(face_data);
    }
}

let video = document.getElementById("myVideo");

let tracker = new tracking.ObjectTracker('face');
tracker.setInitialScale(5);
tracker.setStepSize(1.3);
tracker.setEdgesDensity(0.1);
trackingTask = tracking.track('#myVideo', tracker, {camera:true});
tracker.on('track', function (e) {
    grab_face(e);
});
trackingTask.stop();

function trackFace() {
    trackingTask.run();
    setTimeout(() => {
        const promise = new Promise(function (resolve, reject) {
            trackingTask.stop();
            resolve("ok");
        });
        promise.then(function () {
            let max = 0;
            let emotion = 6;
            for (let i = 0; i < eCount.length; i++) {
                if (eCount[i] > max) {
                    max = eCount[i];
                    emotion = i;
                }
            }
            if (emotion != prevEmotion) {
                prevEmotion = emotion;
                console.log("new emotion is", emotion);
            }
            eCount = [0, 0, 0, 0, 0, 0, 0];
        })
    }, 500);
}

setInterval(trackFace, 500);