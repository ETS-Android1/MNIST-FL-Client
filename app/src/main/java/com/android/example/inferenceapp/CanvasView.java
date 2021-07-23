package com.android.example.inferenceapp;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;


public class CanvasView extends View {
    public static final float TOLERANCE = 5;
    private float mX, mY;
    private Bitmap mBitmap;
    private Canvas mCanvas;
    private final Path mPath;
    private final Paint mPaint;
    Context context;


    public CanvasView(Context context, AttributeSet attrs) {
        super(context, attrs);

        this.context = context;
        mPath = new Path();
        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        mPaint.setColor(Color.BLACK);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mPaint.setStrokeWidth(64f);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        mCanvas = new Canvas(mBitmap);
        mCanvas.drawColor(Color.WHITE);
        this.draw(mCanvas);
    }

    private void touchStart(float x, float y){
        mPath.moveTo(x, y);
        mX = x;
        mY = y;
    }

    private void touchMove(float x, float y){
        float dx = Math.abs(x-mX);
        float dy = Math.abs(y-mY);
        if(dx>=TOLERANCE||dy>=TOLERANCE){
            mPath.quadTo(mX, mY, (x+mX)/2, (y+mY)/2);
            mX = x;
            mY = y;
        }
    }

    private void touchUp(){
        mPath.lineTo(mX, mY);
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();
        switch (event.getAction()){
            case MotionEvent.ACTION_DOWN:
                touchStart(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE:
                touchMove(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                touchUp();
                invalidate();
                break;
        }
        return true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(mPath, mPaint);
    }

    public void clearCanvas(){
        mPath.reset();
        invalidate();
    }

    public void screenShot(String dir) throws FileNotFoundException {
        File full_img = new File(dir, "full_img.jpg");
        File img = new File(dir, "img.jpg");
        Bitmap bmp = getBitmap();
        bmp.compress(Bitmap.CompressFormat.JPEG, 100, new FileOutputStream(full_img));
        Bitmap resized = getResizedBitmap(bmp, 28, 28);
        resized.compress(Bitmap.CompressFormat.JPEG, 100, new FileOutputStream(img));
    }

    public Bitmap getBitmap() {
        this.setDrawingCacheEnabled(true);
        this.draw(mCanvas);
        this.buildDrawingCache();
        Bitmap bmp = Bitmap.createBitmap(this.getDrawingCache());
        this.setDrawingCacheEnabled(false);
        return toGrayscale(bmp);
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }

    public Bitmap getResizedBitmap(Bitmap bm, int newHeight, int newWidth)
    {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false);
    }
}
