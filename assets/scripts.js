$(document).ready(function () {
    $('#lcs_result').hide();
    $("#bt").click(function () {
        // $.get('http://localhost:8007/dl_results', function (data, status) {
        //     console.log(data.result)
        $.post("http://localhost:8007/dl_results",
            {
                title: 'Laptop HP 14-cK0068TU'
                    // $('#title').val()
            },
            function (data, status) {
                console.log( data.result);
            });
    });
});
;


