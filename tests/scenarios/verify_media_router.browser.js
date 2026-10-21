async page => {
  const camera = __CAMERA__;
  const output = __OUTPUT__;
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const card = page.locator('.source-card').filter({hasText:camera});
  await card.locator('img').waitFor({state:'visible'});
  if (!(await card.locator('img').evaluate(image => image.naturalWidth > 0))) throw new Error('Annotated preview not decoded');
  await card.getByRole('button', {name:'Live video',exact:true}).click();
  async function movingVideo() {
    const video = page.frameLocator('#liveFrame').locator('video');
    await video.waitFor();
    return await video.evaluate(async video => {
      const deadline = Date.now() + 25000;
      while ((!video.videoWidth || video.currentTime < .1) && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve,100));
      const before = video.currentTime;
      await new Promise(resolve => setTimeout(resolve,1000));
      if (!video.videoWidth || video.currentTime <= before) throw new Error('Video is not advancing');
      return {width:video.videoWidth,height:video.videoHeight,advance:video.currentTime-before};
    });
  }
  const webrtc = await movingVideo();
  await page.screenshot({path:output+'/dashboard-live.png'});
  await page.locator('#liveProtocol').selectOption('hls');
  const hls = await movingVideo();
  await page.locator('#liveDialog [data-close]').click();
  await page.waitForFunction(() => !document.querySelector('#liveFrame').hasAttribute('src'));
  if (errors.length) throw new Error(JSON.stringify(errors));
  return {passed:true,webrtc,hls,annotatedPreview:true,closeReleasesPlayer:true,errors};
}
