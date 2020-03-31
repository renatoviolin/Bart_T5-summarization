var data = []
var token = ""

jQuery(document).ready(function () {
    var slider = $('#max_words')
    slider.on('change mousemove', function (evt) {
        $('#label_max_words').text('# words in summary: ' + slider.val())
    })

    var slider2 = $('#num_beams')
    slider2.on('change mousemove', function (evt) {
        $('#label_num_beams').text('# beam search: ' + slider2.val())
    })

    $('#btn-process').on('click', function () {
        input_text = $('#txt_input').val()
        model = $('#input_model').val()
        num_words = $('#max_words').val()
        num_beams = $('#num_beams').val()
        $.ajax({
            url: '/predict',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_text": input_text,
                "model": model,
                "num_words": num_words,
                "num_beams": num_beams
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $('#input_summary').val(jsondata['response']['summary'])
        }).fail(function (jsondata, textStatus, jqXHR) {
            alert(jsondata['responseJSON']['message'])
        });
    })


})