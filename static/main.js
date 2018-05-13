let eCount = [0,0,0,0,0,0,0,0];
let prevEmotion = 6;
function countEmotion(emotion) {
    // eCount[emotion] += 1;
    for (let j=0; j<emotion.length; j++){
        for (let i = 0; i < eCount.length-1; i++) {
            eCount[i] += emotion[j][i];
        }
    }
    eCount[7] += 1;
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
    canvas.width = video.offsetWidth;
    canvas.height = video.offsetHeight;
    context.clearRect(0, 0, canvas.width, canvas.height);
    face_data = []
    count = 0
    e.data.forEach(rect => {
        context.drawImage(video, rect.x, rect.y, rect.width+50, rect.height+50, 0, 0, 48, 48);
        data = context.getImageData(0, 0, 48, 48);
        face_data[count] = [];
        // let conv = [0.299, 0.587, 0.114];
        let conv = [0.2126, 0.7152, 0.0722];
        for (let i = 0; i < data.data.length; i += 4) {
            luma = conv[0] * data.data[i] + conv[1] * data.data[i + 1] + conv[2] * data.data[i + 2]
            face_data[count].push(luma);
        }
        count += 1;
    });
    if (count > 0) {
        postData(face_data);
    }
}

let video = document.getElementById("myVideo");
let canvas = document.getElementById("canvas");
let context = canvas.getContext('2d');

let tracker = new tracking.ObjectTracker('face');
tracker.setInitialScale(4.5);
tracker.setStepSize(1.5);
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
            for (let i = 0; i < eCount.length-1; i++) {
                if (eCount[i]/eCount[7] > max) {
                    max = eCount[i];
                    emotion = i;
                }
            }
            if (emotion != prevEmotion) {
                prevEmotion = emotion;
                console.log("new emotion is", emotion);
            }
            eCount = [0, 0, 0, 0, 0, 0, 0, 0];
        })
    }, 100);
}

setInterval(trackFace, 1000);