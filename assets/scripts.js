$(document).ready(function () {
    $('#lcs_result').hide();
    $("#bt").click(function () {
        // $.get('http://localhost:8007/dl_results', function (data, status) {
        //     console.log(data.result)
        $.post("http://localhost:8007/dl_results",
            {
                "title": $('#title').val()
                    // $('#title').val()
            },
            function (data, status) {
                console.log( data.result);
                var i=0
                var res='lcs results:<br>'
                for (i=0; i<10; i++){
                    res+=data.result[i]+'<br>'
                }
                $('#lcs_result').html()=res
                $('#lcs_result').show()
            });
    });
});
;


