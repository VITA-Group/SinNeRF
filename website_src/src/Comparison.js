import { createMedia } from "@artsy/fresnel";
import PropTypes from "prop-types";
import "semantic-ui-css/semantic.css";
import React, { Component } from "react";
import subsection_style from "./style.js";

import LazyVideo from "./lazy-video";

import {
  Button,
  Container,
  Divider,
  Grid,
  Header,
  Icon,
  Image,
  List,
  Menu,
  Segment,
  Sidebar,
  Visibility,
} from "semantic-ui-react";

import flower_diet from "./video/flower_diet.mp4";
import flower_dsnerf from "./video/flower_dsnerf.mp4";
import flower_ours from "./video/flower_ours.mp4";
import flower_pix from "./video/flower_pix.mp4";
import fortress_diet from "./video/fortress_diet.mp4";
import fortress_dsnerf from "./video/fortress_dsnerf.mp4";
import fortress_ours from "./video/fortress_ours.mp4";
import fortress_pix from "./video/fortress_pix.mp4";
import hotdog_diet from "./video/hotdog_diet.mp4";
import hotdog_dsnerf from "./video/hotdog_dsnerf.mp4";
import hotdog_ours from "./video/hotdog_ours.mp4";
import hotdog_pix from "./video/hotdog_pix.mp4";
import lego_diet from "./video/lego_diet.mp4";
import lego_dsnerf from "./video/lego_dsnerf.mp4";
import lego_ours from "./video/lego_ours.mp4";
import lego_pix from "./video/lego_pix.mp4";
import room_dietnerf from "./video/room_dietnerf.mp4";
import room_dsnerf from "./video/room_dsnerf.mp4";
import room_ours from "./video/room_ours.mp4";
import room_pix from "./video/room_pix.mp4";

const myVideo = () => (
  <div>
    <Header as="h2" textAlign="center" style={subsection_style}>
      Comparisons
    </Header>
    <p>
      We compare our method with the state-of-the-art neural radiance field
      methods DietNeRF, PixelNeRF and DS-NeRF. Our method generates the most
      visually-pleasing results, while other methods tend to render obscure
      estimations on novel views. DS-NeRF shows realistic geometry but the
      rendered images are blurry. PixelNeRF and DietNeRF present good structures
      but wrong geometry due to their lack of local texture guidance and
      geometry pseudo label.
    </p>
    <Grid columns={4}>
      <Grid.Row textAlign="center" style={{ fontSize: "1.2em" }}>
        <Grid.Column>DS-NeRF</Grid.Column>
        <Grid.Column>DietNeRF</Grid.Column>
        <Grid.Column>PixelNeRF</Grid.Column>
        <Grid.Column>
          <b>SinNeRF</b>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={flower_dsnerf}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={flower_diet}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={flower_pix}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={flower_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={room_dsnerf}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={room_dietnerf}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={room_pix}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={room_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={fortress_dsnerf}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={fortress_diet}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={fortress_pix}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={fortress_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={lego_dsnerf}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={lego_diet}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={lego_pix}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={lego_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={hotdog_dsnerf}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={hotdog_diet}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={hotdog_pix}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={hotdog_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
    </Grid>
  </div>
);

export default myVideo;
