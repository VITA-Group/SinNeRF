import { createMedia } from "@artsy/fresnel";
import PropTypes from "prop-types";
import "semantic-ui-css/semantic.css";
import React, { Component } from "react";
import subsection_style from "./style.js";

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

import scan3_ours from "./video/scan3_ours.mp4";
import scan4_ours from "./video/scan4_ours.mp4";
import scan5_ours from "./video/scan5_ours.mp4";
import scan34_ours from "./video/scan34_ours.mp4";
import scan40_ours from "./video/scan40_ours.mp4";
import scan60_ours from "./video/scan60_ours.mp4";
import scan63_ours from "./video/scan63_ours.mp4";
import scan82_ours from "./video/scan82_ours.mp4";
import scan84_ours from "./video/scan84_ours.mp4";

import LazyVideo from "./lazy-video";

const myVideo = () => (
  <div>
    <Header as="h2" textAlign="center" style={subsection_style}>
      Our Results
    </Header>
    <Grid columns={2}>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={lego_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={hotdog_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
    </Grid>
    <Grid columns={3}>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={flower_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={room_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={fortress_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={scan3_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={scan4_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={scan5_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={scan34_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={scan40_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={scan60_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
      <Grid.Row>
        <Grid.Column>
          <LazyVideo src={scan63_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={scan82_ours}></LazyVideo>
        </Grid.Column>
        <Grid.Column>
          <LazyVideo src={scan84_ours}></LazyVideo>
        </Grid.Column>
      </Grid.Row>
    </Grid>
  </div>
);

export default myVideo;
