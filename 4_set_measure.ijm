run("Measure...", "choose=/home/gabri/Desktop/obj-det/models/object_detection/immagini_da_estrarre_roi/output/roi_norm/");
saveAs("Results", "/home/gabri/Desktop/obj-det/models/object_detection/immagini_da_estrarre_roi/output/measures.csv");

macro "Batch GLCM Measure" {
    dir = "/home/gabri/Desktop/obj-det/models/object_detection/immagini_da_estrarre_roi/output/roi_norm/";
    list = getFileList(dir);
    step = 1;
    setBatchMode(true);
    setOption("ExpandableArrays", true);
    /* print("#,","Angular Second Moment,","Contrast,","Correlation,","Inverse Difference Moment,","Entropy,"); */
    numArr = newArray("#");
    asmArr = newArray("Angular Second Moment");
    conArr = newArray("Contrast");
    corArr = newArray("Correlation");
    idmArr = newArray("Inverse Difference Moment");
    entArr = newArray("Entropy");
    /* numArr[0] = "#"; */
    /* asmArr[0] =  */
    /* conArr[0] = "Contrast"; */
    /* corArr[0] = "Correlation"; */
    /* idmArr[0] = "Inverse Difference Moment"; */
    /* entArr[0] = "Entropy"; */

    /* headArr = newArray("#,","Angular Second Moment,","Contrast,","Correlation,","Inverse Difference Moment,","Entropy,"); */
    
    for (i=0; i<list.length; i++) {
        path = dir+list[i];
        showProgress(i, list.length);
        if (!endsWith(path,"/")) open(path);
        if (nImages>=1) {
	    run("8-bit");
	    run("GLCM Texture", "enter=1 select=[0 degrees] angular contrast correlation inverse entropy");
            // run("GLCM Texture", "enter="+step+ " select=[0 degrees] angular contrast correlation inverse entropy");
            close();
            asmArr[i] = getResult("Angular Second Moment",0); 
            conArr[i] = getResult("Contrast",0);
            corArr[i] = getResult("Correlation",0);
            idmArr[i] = getResult("Inverse Difference Moment   ",0); //Extra spaces needed due to source code error
            entArr[i] = getResult("Entropy",0);
	    numArr[i] = list[i];
            /* print(list[i],",",asm,",",contrast,",",correlation,",",idm,",",entropy); */
	    /* setResult("#",list[i],i); */
	    /* setResult("Angular Second Moment",asm,i); */
            /* Array.show(list[i],",",asm,",",contrast,",",correlation,",",idm,",",entropy); */
        }
    }
    /* selectWindow("Log");  //select Log-window  */
    /* saveAs("Text", "/home/gabri/Desktop/obj-det/models/object_detection/immagini_da_estrarre_roi/output/GLCM.xls"); */
run("Clear Results");
for (i=0; i<asmArr.length; i++){
    setResult("#",i,numArr[i]);
    setResult("Angular Second Moment",i,asmArr[i]);
    setResult("Contrast",i,conArr[i]);
    setResult("Correlation",i,corArr[i]);
    setResult("Inverso Difference Moment",i,idmArr[i]);
    setResult("Entropy",i,entArr[i]);
}
updateResults;
    /* selectWindow("Results");  //select Log-window  */
    /* saveAs("Results", "/home/gabri/Desktop/obj-det/models/object_detection/immagini_da_estrarre_roi/output/GLCM.xls"); */

outFile = File.open("/home/gabri/Desktop/obj-det/models/object_detection/immagini_da_estrarre_roi/output/GLCM.txt") ; // create results file
head = "#,Angular Second Moment,Contrast,Correlation,Inverse Difference Moment,Entropy,";
print(outFile, head);
for (i=0; i<asmArr.length; i++)                        // loop through well results
{
    /* line = (list[i]+","+asmArr[i]+","+conArr[i]+",",corArr[i],",",idmArr[i],",",ent[i]);	 */
    line = list[i]+","+asmArr[i]+","+conArr[i]+","+corArr[i]+","+idmArr[i]+","+entArr[i];	
    print(outFile, line);
/* dataRow = asmArr[j];             // assign current well results */
/* print(outFile, dataRow);              // write current well results  */
}
File.close(outFile); 
}




