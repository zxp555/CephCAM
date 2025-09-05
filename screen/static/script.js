function full_screen() {
    const elem = document.getElementById('content');
    if (elem.requestFullscreen) {
        elem.requestFullscreen();
    } else if (elem.mozRequestFullScreen) { // Firefox
        elem.mozRequestFullScreen();
    } else if (elem.webkitRequestFullscreen) { // Chrome, Safari and Opera
        elem.webkitRequestFullscreen();
    } else if (elem.msRequestFullscreen) { // IE/Edge
        elem.msRequestFullscreen();
    }
}

function set_text(str) {
    document.getElementById("text").innerHTML = str
}

function connect() {
    let eventSource = new EventSource('/inform');
    eventSource.onmessage = function(event) {
        let data = JSON.parse(event.data);
        type = data['t']
        data = data['d']
        console.log(type, data)

        if (type === 'hello') {
            set_text(`Connected<br>R${data}<br>${screen.width}*${screen.height}`)
        }
        else if (type === 'img') {
            document.getElementById('image').src = data;
        }
        else {
            console.log("???")
        }
    };
    eventSource.onerror = function(event) {
        console.error("SSE 连接发生错误:", event);
        document.getElementById('image').src = '';
        set_text('Fail');
        eventSource.close();
        setTimeout(connect, 1000);
    };
}

function main() {
    document.getElementById('start').remove();
    full_screen();
    setTimeout(connect, 1000);
}

document.getElementById('start').addEventListener('click', main);